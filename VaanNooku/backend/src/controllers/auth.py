from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from jose import jwt
from datetime import datetime, timedelta
from src.models import models
from src.schemas import schemas

JWT_SECRET = "retail_ai_secret_key_9988"
ALGORITHM = "HS256"

def login_store(db: Session, request: schemas.LoginRequest):
    # Find store matching username/name
    store = db.query(models.Store).filter(
        (models.Store.name == request.username) | 
        (models.Store.id == request.username.lower().replace(" ", "-"))
    ).first()

    if not store:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Store credentials not found."
        )

    # Issue access token
    access_token = jwt.encode(
        {"sub": store.id, "exp": datetime.utcnow() + timedelta(days=7)},
        JWT_SECRET,
        algorithm=ALGORITHM
    )

    return {
        "token": access_token,
        "store": {
            "id": store.id,
            "name": store.name,
            "type": store.type,
            "location": store.location,
            "investment": store.investment,
            "openingMonth": store.opening_month,
            "monthsActive": store.months_active,
            "metrics": {
                "forecastR2": 0.932,
                "wasteMargin": 0.024,
                "stockouts": 2,
                "deficitCount": 3
            }
        }
    }

def register_store(db: Session, request: schemas.StoreRegisterRequest):
    store_id = request.storeName.lower().strip().replace(" ", "-").replace("/", "-")
    
    # Check duplicate
    existing = db.query(models.Store).filter(models.Store.id == store_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A store with this name is already registered."
        )

    # Insert Store Node
    new_store = models.Store(
        id=store_id,
        name=request.storeName,
        type=request.storeType,
        location=request.locationType,
        investment=request.investment,
        opening_month=request.openingMonth,
        months_active=12
    )
    db.add(new_store)
    db.commit()

    # Map Supplier
    supplier_id = "balaji-agro"
    if request.supplier and request.supplier.name:
        supplier_id = request.supplier.name.lower().strip().replace(" ", "-")
        # Check if supplier already exists
        sup_exists = db.query(models.Supplier).filter(models.Supplier.id == supplier_id).first()
        if not sup_exists:
            new_sup = models.Supplier(
                id=supplier_id,
                name=request.supplier.name,
                category="General Wholesale",
                phone=request.supplier.phone,
                email=request.supplier.email
            )
            db.add(new_sup)
            db.commit()

    # Ingest catalog products
    if request.productsList:
        for p in request.productsList:
            prod_id = p.name.lower().strip().replace(" ", "-").replace("/", "-")
            
            # Map cost and price
            cost = p.buyingPrice or p.cost_price or 10.0
            price = p.sellingPrice or p.price or 12.0
            
            # Insert product to general catalog if not exists
            prod_exists = db.query(models.Product).filter(models.Product.id == prod_id).first()
            if not prod_exists:
                new_prod = models.Product(
                    id=prod_id,
                    name=p.name,
                    sku="SKU-" + prod_id.upper()[:8],
                    category=p.category or "General",
                    cost_price=cost,
                    selling_price=price,
                    supplier_id=supplier_id
                )
                db.add(new_prod)
                db.commit()

            # Map stock level
            new_stock = models.StockLevel(
                store_id=store_id,
                product_id=prod_id,
                current_stock=p.qty,
                safety_stock=15,
                reorder_point=30
            )
            db.add(new_stock)
            db.commit()
    else:
        # Default seeds
        default_prods = db.query(models.Product).limit(5).all()
        for dp in default_prods:
            new_stock = models.StockLevel(
                store_id=store_id,
                product_id=dp.id,
                current_stock=100,
                safety_stock=15,
                reorder_point=30
            )
            db.add(new_stock)
            db.commit()

    return {
        "success": True,
        "storeId": store_id,
        "message": "Store successfully registered and inventory initialized."
    }
