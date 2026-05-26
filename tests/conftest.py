import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

os.environ["DATABASE_URL"] = "sqlite:///./test_prism.db"

from db.database import init_db
init_db()