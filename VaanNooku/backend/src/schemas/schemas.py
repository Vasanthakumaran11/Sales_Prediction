from pydantic import BaseModel, Field
from typing import List, Optional

class SupplierBase(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None

class ProductBase(BaseModel):
    name: str
    qty: int
    buyingPrice: Optional[float] = None
    cost_price: Optional[float] = None
    sellingPrice: Optional[float] = None
    price: Optional[float] = None
    category: Optional[str] = None

class StoreRegisterRequest(BaseModel):
    storeName: str
    storeType: Optional[str] = "Supermarket"
    locationType: Optional[str] = "Urban"
    openingMonth: Optional[str] = "October"
    investment: float
    productsList: Optional[List[ProductBase]] = []
    supplier: Optional[SupplierBase] = None

class LoginRequest(BaseModel):
    username: str
    password: Optional[str] = None

class PaymentDetails(BaseModel):
    cash: float = 0.0
    upi: float = 0.0
    card: float = 0.0

class DailyLogItem(BaseModel):
    productId: str
    qty: int
    discount: float = 0.0

class DailyLogSubmitRequest(BaseModel):
    date: str
    paymentDetails: Optional[PaymentDetails] = None
    items: List[DailyLogItem]
    notes: Optional[str] = None

class SkuAddRequest(BaseModel):
    sku: str
    name: str
    category: Optional[str] = "General"
    costPrice: float
    sellingPrice: float

class POItem(BaseModel):
    name: str
    qty: int
    cost: float

class POOrderRequest(BaseModel):
    items: List[POItem]

class ProfileUpdateRequest(BaseModel):
    name: str
    email: str
    phone: str

class PasswordUpdateRequest(BaseModel):
    currentPassword: str
    newPassword: str
