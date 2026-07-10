from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile, status
import pandas as pd
import io
import time
from src.models import models
from src.schemas import schemas

def submit_log(db: Session, store_id: str, request: schemas.DailyLogSubmitRequest):
    # Verify store
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store node not found")

    transaction_id = "TX-" + str(int(time.time()))
    gross_sales = 0.0
    discount_amount = 0.0

    for item in request.items:
        # Find product matching ID or name
        prod = db.query(models.Product).filter(
            (models.Product.id == item.productId) | 
            (models.Product.name == item.productId)
        ).first()

        if not prod:
            continue

        item_cost = prod.selling_price * item.qty
        gross_sales += item_cost
        discount_amount += item.discount

        # Insert Transaction Item
        tx_item = models.TransactionItem(
            id="TI-" + str(int(time.time())) + "-" + prod.id[:3],
            transaction_id=transaction_id,
            product_id=prod.id,
            qty=item.qty,
            discount=item.discount
        )
        db.add(tx_item)

        # Decrement Inventory stock level
        stock_level = db.query(models.StockLevel).filter(
            models.StockLevel.store_id == store_id,
            models.StockLevel.product_id == prod.id
        ).first()

        if stock_level:
            stock_level.current_stock = max(0, stock_level.current_stock - item.qty)

    net_sales = gross_sales - discount_amount

    # Save daily log summary
    daily_log = models.TransactionDaily(
        id=transaction_id,
        store_id=store_id,
        date=request.date,
        transaction_count=len(request.items),
        gross_sales=gross_sales,
        discount_amount=discount_amount,
        net_sales=net_sales,
        payment_cash=request.paymentDetails.cash if request.paymentDetails else 0.0,
        payment_upi=request.paymentDetails.upi if request.paymentDetails else 0.0,
        payment_card=request.paymentDetails.card if request.paymentDetails else 0.0,
        audit_status="Synced & Closed"
    )
    db.add(daily_log)
    db.commit()

    return {
        "success": True,
        "logId": transaction_id,
        "netSales": net_sales
    }

def get_logs(db: Session, store_id: str):
    logs = db.query(models.TransactionDaily).filter(
        models.TransactionDaily.store_id == store_id
    ).order_by(models.TransactionDaily.date.desc()).all()
    return logs

def parse_sales_csv(file: UploadFile):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Simulate loading products from csv upload
        parsed_products = [
            { "id": 201, "name": "Premium Kolam Rice 10kg", "category": "Staples & Grains", "buyingPrice": 580.0, "sellingPrice": 650.0, "qty": 100, "checked": True },
            { "id": 202, "name": "Madhur Pure Sugar 5kg", "category": "Staples & Grains", "buyingPrice": 210.0, "sellingPrice": 240.0, "qty": 150, "checked": True },
            { "id": 203, "name": "Britannia Marie Gold 250g", "category": "Snacks & Biscuits", "buyingPrice": 30.0, "sellingPrice": 35.0, "qty": 500, "checked": True },
            { "id": 204, "name": "Amul Pure Ghee 1L", "category": "Dairy & Bakery", "buyingPrice": 610.0, "sellingPrice": 650.0, "qty": 60, "checked": True }
        ]
        
        return {
            "itemsCount": len(parsed_products),
            "products": parsed_products
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Spreadsheet parser failure: {str(err)}"
        )
