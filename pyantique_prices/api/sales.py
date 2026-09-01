"""Historical sales endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from pyantique_prices.data.models import HistoricalSale

router = APIRouter()


@router.get("/sales/{sale_id}")
def get_sale(sale_id: int, request: Request) -> dict:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        sale = session.query(HistoricalSale).filter_by(id=sale_id).first()
        if sale is None:
            raise HTTPException(status_code=404, detail="Sale not found")
        return {
            "id": sale.id,
            "title": sale.title,
            "description": sale.description,
            "object_type": sale.object_type,
            "manufacturer": sale.manufacturer,
            "country": sale.country,
            "sale_date": sale.sale_date.isoformat() if sale.sale_date else None,
            "currency": sale.currency,
            "final_price": sale.final_price,
            "normalized_currency": sale.normalized_currency,
            "normalized_price": sale.normalized_price,
            "source_url": sale.source_url,
        }
