from sqlalchemy.orm import Session
from fastapi import HTTPException
import time
from src.models import models
from src.schemas import schemas

def get_inventory(db: Session, store_id: str):
    results = db.query(
        models.Product.id,
        models.Product.name,
        models.Product.sku,
        models.Product.category,
        models.Product.cost_price,
        models.Product.selling_price,
        models.StockLevel.current_stock,
        models.StockLevel.safety_stock,
        models.StockLevel.reorder_point
    ).join(
        models.StockLevel, models.Product.id == models.StockLevel.product_id
    ).filter(
        models.StockLevel.store_id == store_id
    ).all()

    inventory = []
    for row in results:
        margin = (row.selling_price - row.cost_price) / row.selling_price if row.selling_price > 0 else 0
        inventory.append({
            "id": row.id,
            "name": row.name,
            "sku": row.sku,
            "category": row.category,
            "cost": row.cost_price,
            "price": row.selling_price,
            "margin": round(margin * 100, 1),
            "stock": row.current_stock,
            "minStock": row.safety_stock,
            "rop": row.reorder_point,
            "eoq": 150
        })

    return inventory

def get_bulk_consolidation(db: Session):
    # Fetch shortages items where current stock <= ROP
    results = db.query(
        models.Product.id,
        models.Product.name,
        models.Product.category,
        models.Product.cost_price,
        models.StockLevel.current_stock,
        models.StockLevel.reorder_point
    ).join(
        models.StockLevel, models.Product.id == models.StockLevel.product_id
    ).filter(
        models.StockLevel.current_stock <= models.StockLevel.reorder_point
    ).all()

    items_map = {}
    for row in results:
        required = row.reorder_point - row.current_stock
        if row.name in items_map:
            items_map[row.name]["totalQty"] += required
            items_map[row.name]["standardCost"] += (required * row.cost_price)
        else:
            items_map[row.name] = {
                "name": row.name,
                "price": row.cost_price,
                "totalQty": required,
                "standardCost": required * row.cost_price
            }

    items = list(items_map.values())
    
    # Ensure minimum quantity of 20 for bulk contract if empty
    if not items:
        items = [{
            "name": "Aashirvaad Chakki Atta 5kg",
            "price": 245.0,
            "totalQty": 50,
            "standardCost": 12250.0
        }]

    total_standard_cost = sum(i["standardCost"] for i in items)
    discount_rate = 0.15
    total_discounted_cost = int(total_standard_cost * (1 - discount_rate))

    return {
        "items": items,
        "discountRate": discount_rate,
        "totalStandardCost": total_standard_cost,
        "totalDiscountedCost": total_discounted_cost,
        "totalSavings": total_standard_cost - total_discounted_cost
    }

def place_bulk_order(request: schemas.POOrderRequest):
    return {
        "orderId": f"PO-{int(time.time())}",
        "status": "dispatched",
        "estimatedDelivery": "2026-05-20",
        "items": [item.dict() for item in request.items]
    }

def get_products(db: Session, store_id: str, category: str = None, search: str = None):
    query = db.query(models.Product, models.StockLevel.current_stock).join(
        models.StockLevel, 
        (models.Product.id == models.StockLevel.product_id) & (models.StockLevel.store_id == store_id),
        isouter=True
    )

    if category:
        query = query.filter(models.Product.category == category)
    if search:
        query = query.filter(
            (models.Product.name.like(f"%{search}%")) | 
            (models.Product.sku.like(f"%{search}%"))
        )

    results = query.all()
    products = []
    for prod, stock in results:
        margin = (prod.selling_price - prod.cost_price) / prod.selling_price if prod.selling_price > 0 else 0
        products.append({
            "id": prod.id,
            "name": prod.name,
            "sku": prod.sku,
            "category": prod.category,
            "stock": stock if stock is not None else 0,
            "cost": prod.cost_price,
            "price": prod.selling_price,
            "margin": round(margin * 100, 1),
            "status": "Healthy" if (stock or 0) > 30 else "Low Stock"
        })

    return products

def add_product(db: Session, store_id: str, request: schemas.SkuAddRequest):
    prod_id = request.name.lower().strip().replace(" ", "-").replace("/", "-")
    
    # Check duplicate SKU
    exists = db.query(models.Product).filter(models.Product.sku == request.sku).first()
    if exists:
        raise HTTPException(status_code=409, detail="Product barcode SKU already registered.")

    new_prod = models.Product(
        id=prod_id,
        name=request.name,
        sku=request.sku,
        category=request.category,
        cost_price=request.costPrice,
        selling_price=request.sellingPrice
    )
    db.add(new_prod)
    db.commit()

    # Map inventory connection
    new_stock = models.StockLevel(
        store_id=store_id,
        product_id=prod_id,
        current_stock=50,
        safety_stock=10,
        reorder_point=20
    )
    db.add(new_stock)
    db.commit()

    return {"success": True, "productId": prod_id}
