"""
Per-participant naming for the hands-on lab.

This is a shared-cluster workshop: many participants run the same code at the
same time against one Hive Metastore. Every object therefore has to be namespaced
per user, derived the same way in every script, and safe to re-run. Deriving the
names in one place is what makes that true -- when 01_ingest.py, the EDA notebook
and reset.py each build the name with their own f-string, they drift, and the
symptom is a participant whose data "disappeared".

Names are lowercase throughout. Hive stores identifiers lowercase regardless of
how you type them, so lowercasing here keeps the SQL name and the HDFS directory
visibly identical instead of leaving you to reconcile DEFAULT_OZARATE with
.../default_ozarate.db.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

# Database prefix. Descriptive on purpose: these databases sit next to
# production ones in the metastore, and a name like "default_<user>" invites
# confusion with Hive's real `default` database.
WORKSHOP_PREFIX = "ml_workshop"

TABLE_NAME = "bank_marketing"

# CDP's conventional external warehouse path, used only if the site's own
# configuration cannot be read.
_FALLBACK_EXTERNAL_WAREHOUSE = "/warehouse/tablespace/external/hive"

_HIVE_SITE_GLOBS = (
    "/var/lib/cdsw/client-config/hive-conf/hive-site.xml",
    "/var/lib/cdsw/client-config/*/hive-site.xml",
    "/etc/hive/conf/hive-site.xml",
)


def raw_username():
    """The workshop participant's identity, as the platform reports it."""
    return os.environ.get("HADOOP_USER_NAME") or os.environ.get("PROJECT_OWNER") or ""


def safe_username(username=None):
    """
    Reduce a username to something legal as a Hive identifier and a path segment.

    Real workshop rosters contain names like 'jim.right', 'o'brien' or
    'ozarate@EXAMPLE.COM'. Interpolated raw into SQL, 'ml_workshop_jim.right'
    parses as database `ml_workshop_jim`, table `right` -- so the script appears
    to work while writing somewhere nobody expects. Sanitising is not cosmetic.
    """
    name = (username or raw_username()).strip().lower()
    name = name.split("@", 1)[0]                 # drop any Kerberos realm
    name = re.sub(r"[^a-z0-9_]+", "_", name)     # only identifier-safe chars
    name = re.sub(r"_+", "_", name).strip("_")   # tidy up runs of underscores
    if not name:
        raise RuntimeError(
            "Could not determine a usable username. Expected HADOOP_USER_NAME or "
            "PROJECT_OWNER to be set in this session."
        )
    if not re.match(r"^[a-z_]", name):
        name = f"u_{name}"                       # identifiers cannot start with a digit
    return name


def workshop_database(username=None):
    """e.g. 'ml_workshop_ozarate' -- one database per participant."""
    return f"{WORKSHOP_PREFIX}_{safe_username(username)}"


def workshop_table(username=None):
    """The table name. Unqualified: the database is already per-participant."""
    return TABLE_NAME


def workshop_table_path(username=None):
    """Fully-qualified 'database.table' for use in Spark SQL."""
    return f"{workshop_database(username)}.{workshop_table(username)}"


def external_warehouse_dir():
    """
    The site's external warehouse root.

    Resolution order:
      1. WORKSHOP_WAREHOUSE_DIR -- explicit admin override, wins over everything.
      2. hive.metastore.warehouse.external.dir from the cluster's own client
         config. This is the portable option: it reads s3a://... on public cloud
         and hdfs paths on prem, so the same code works in both.
      3. CDP's conventional path.

    The *external* warehouse is deliberate. A database created with no LOCATION
    inherits hive.metastore.warehouse.dir -- the managed, ACID-controlled
    tablespace, which Ranger commonly restricts to the Hive service user. Twenty
    participants each trying to create a database there is a mass failure at the
    first step of the lab.
    """
    override = os.environ.get("WORKSHOP_WAREHOUSE_DIR", "").strip()
    if override:
        return override.rstrip("/")

    for pattern in _HIVE_SITE_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                root = ET.parse(path).getroot()
            except (ET.ParseError, OSError):
                continue
            for prop in root.findall("property"):
                if prop.findtext("name") == "hive.metastore.warehouse.external.dir":
                    value = (prop.findtext("value") or "").strip()
                    if value:
                        return value.rstrip("/")

    return _FALLBACK_EXTERNAL_WAREHOUSE


def workshop_database_location(username=None):
    """Explicit LOCATION for this participant's database."""
    return f"{external_warehouse_dir()}/{workshop_database(username)}.db"


def describe(username=None):
    """One-line summary, so every script logs where it is about to write."""
    return (f"database={workshop_database(username)} "
            f"table={workshop_table(username)} "
            f"location={workshop_database_location(username)}")
