"""add_user_rating_column_to_interactions

Revision ID: facf86dedb69
Revises: 65ba8a38a7a7
Create Date: 2025-08-19 01:35:57.890236

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'facf86dedb69'
down_revision: Union[str, Sequence[str], None] = '65ba8a38a7a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add user_rating column to interactions table
    op.add_column('interactions', sa.Column('user_rating', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove user_rating column from interactions table
    op.drop_column('interactions', 'user_rating')
