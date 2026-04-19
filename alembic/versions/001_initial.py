"""Initial schema: 13 entities for passenger forecast IS.

Revision ID: 001_initial
Revises:
Create Date: 2026-04-19
"""

from alembic import op


revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the DDL schema from db/schema.sql."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "db", "schema.sql")
    schema_path = os.path.abspath(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        op.execute(f.read())


def downgrade() -> None:
    op.execute("""
        DROP TABLE IF EXISTS trip_factor CASCADE;
        DROP TABLE IF EXISTS passenger_count CASCADE;
        DROP TABLE IF EXISTS trip CASCADE;
        DROP TABLE IF EXISTS external_factor CASCADE;
        DROP TABLE IF EXISTS model_metrics CASCADE;
        DROP TABLE IF EXISTS forecast CASCADE;
        DROP TABLE IF EXISTS forecast_model CASCADE;
        DROP TABLE IF EXISTS report CASCADE;
        DROP TABLE IF EXISTS app_user CASCADE;
        DROP TABLE IF EXISTS route_station CASCADE;
        DROP TABLE IF EXISTS route CASCADE;
        DROP TABLE IF EXISTS station CASCADE;
        DROP TABLE IF EXISTS region CASCADE;
    """)
