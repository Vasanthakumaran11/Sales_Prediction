from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from src.models import models
from src.schemas import schemas

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

    if store.password and store.password != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect store password."
        )

    return {
        "token": "auth-token-disabled",
        "store": {
            "id": store.id,
            "name": store.name,
            "type": store.type,
            "location": store.location,
            "investment": store.investment,
            "openingMonth": store.opening_month,
            "monthsActive": store.months_active,
            "adminName": store.admin_name,
            "adminEmail": store.admin_email,
            "adminPhone": store.admin_phone,
            "adminRole": store.admin_role,
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
        password=request.password,
        type=request.storeType,
        location=request.locationType,
        investment=request.investment,
        opening_month=request.openingMonth,
        months_active=12,
        admin_name=request.adminName,
        admin_email=request.adminEmail,
        admin_phone=request.adminPhone,
        admin_role=request.adminRole or "Store Owner"
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

    # Ingest catalog products dynamically if provided
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

    return {
        "id": new_store.id,
        "name": new_store.name,
        "type": new_store.type,
        "location": new_store.location,
        "investment": new_store.investment,
        "openingMonth": new_store.opening_month,
        "monthsActive": new_store.months_active,
        "adminName": new_store.admin_name,
        "adminEmail": new_store.admin_email,
        "adminPhone": new_store.admin_phone,
        "adminRole": new_store.admin_role,
        "metrics": {
            "forecastR2": 0.90,
            "wasteMargin": 0.05,
            "stockouts": 0,
            "deficitCount": 0
        }
    }
