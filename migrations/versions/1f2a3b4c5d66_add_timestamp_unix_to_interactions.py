"""add timestamp_unix to interactions and backfill

Revision ID: 1f2a3b4c5d66
Revises: 8ce751882688
Create Date: 2025-08-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1f2a3b4c5d66'
down_revision: Union[str, Sequence[str], None] = '8ce751882688'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by adding timestamp_unix and backfilling from timestamp."""
    # Add the column as nullable to allow backfill without constraint violations
    op.add_column('interactions', sa.Column('timestamp_unix', sa.Integer(), nullable=True))

    # Backfill using SQLite strftime to compute Unix seconds from datetime
    # Works for SQLite; for other DBs it's a no-op or will need dialect-specific migration
    conn = op.get_bind()
    # Update only rows where timestamp is not null
    conn.exec_driver_sql(
        """
        UPDATE interactions
        SET timestamp_unix = CAST(strftime('%s', timestamp) AS INTEGER)
        WHERE timestamp IS NOT NULL AND timestamp_unix IS NULL
        """
    )

    # Optionally, make it non-nullable for new rows if you need strictness
    # Keeping it nullable in case some rows legitimately lack timestamp
    # op.alter_column('interactions', 'timestamp_unix', existing_type=sa.Integer(), nullable=False)


def downgrade() -> None:
    """Downgrade schema by dropping timestamp_unix."""
    op.drop_column('interactions', 'timestamp_unix')


