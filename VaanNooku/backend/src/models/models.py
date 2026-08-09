from sqlalchemy import Column, String, Integer, Float, ForeignKey, Date, Enum, Table, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from src.config.database import Base

class Store(Base):
    __tablename__ = "stores"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    password = Column(String, nullable=True)
    type = Column(String, default="Supermarket") # Small | Medium | Supermarket
    location = Column(String, default="Urban") # Urban | Semi-Urban | Rural
    investment = Column(Float, nullable=False)
    opening_month = Column(String, nullable=False)
    months_active = Column(Integer, default=12)

    # Admin / Owner contact details
    admin_name = Column(String, nullable=True)
    admin_email = Column(String, nullable=True)
    admin_phone = Column(String, nullable=True)
    admin_role = Column(String, nullable=True, default="Store Owner")

    # Relationships
    stock_levels = relationship("StockLevel", back_populates="store", cascade="all, delete-orphan")
    transactions = relationship("TransactionDaily", back_populates="store", cascade="all, delete-orphan")


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    lead_time_days = Column(Integer, default=3)
    min_order_qty = Column(Integer, default=10)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)

    products = relationship("Product", back_populates="supplier")


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    sku = Column(String, unique=True, index=True, nullable=False)
    category = Column(String, nullable=False)
    brand = Column(String, nullable=True)
    cost_price = Column(Float, nullable=False)
    selling_price = Column(Float, nullable=False)
    supplier_id = Column(String, ForeignKey("suppliers.id", ondelete="SET NULL"), nullable=True)

    # Relationships
    supplier = relationship("Supplier", back_populates="products")
    stock_levels = relationship("StockLevel", back_populates="product", cascade="all, delete-orphan")
    transaction_items = relationship("TransactionItem", back_populates="product", cascade="all, delete-orphan")


class StockLevel(Base):
    __tablename__ = "stock_levels"

    store_id = Column(String, ForeignKey("stores.id", ondelete="CASCADE"), primary_key=True)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), primary_key=True)
    current_stock = Column(Integer, default=0)
    safety_stock = Column(Integer, default=15)
    reorder_point = Column(Integer, default=30)

    # Relationships
    store = relationship("Store", back_populates="stock_levels")
    product = relationship("Product", back_populates="stock_levels")


class TransactionDaily(Base):
    __tablename__ = "transactions_daily"

    id = Column(String, primary_key=True, index=True)
    store_id = Column(String, ForeignKey("stores.id", ondelete="CASCADE"), nullable=False)
    date = Column(String, nullable=False)
    transaction_count = Column(Integer, default=0)
    gross_sales = Column(Float, default=0.0)
    discount_amount = Column(Float, default=0.0)
    net_sales = Column(Float, default=0.0)
    payment_cash = Column(Float, default=0.0)
    payment_upi = Column(Float, default=0.0)
    payment_card = Column(Float, default=0.0)
    audit_status = Column(String, default="Synced & Closed") # Synced & Closed | Draft

    # Relationships
    store = relationship("Store", back_populates="transactions")
    items = relationship("TransactionItem", back_populates="transaction", cascade="all, delete-orphan")


class TransactionItem(Base):
    __tablename__ = "transaction_items"

    id = Column(String, primary_key=True, index=True)
    transaction_id = Column(String, ForeignKey("transactions_daily.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    qty = Column(Integer, nullable=False)
    discount = Column(Float, default=0.0)

    # Relationships
    transaction = relationship("TransactionDaily", back_populates="items")
    product = relationship("Product", back_populates="transaction_items")


class AdminUser(Base):
    """Ops/staff console account — intentionally separate from Store so a
    merchant JWT can never be used against admin-only endpoints."""
    __tablename__ = "admin_users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="admin")  # admin | ml_engineer | support


class ComplaintTicket(Base):
    __tablename__ = "complaint_tickets"

    id = Column(String, primary_key=True, index=True)
    store_id = Column(String, ForeignKey("stores.id", ondelete="CASCADE"), nullable=False)
    subject = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status = Column(String, default="Open")  # Open | In Progress | Resolved
    created_at = Column(String, nullable=False)
