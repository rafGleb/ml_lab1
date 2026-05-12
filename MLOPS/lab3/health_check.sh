#!/bin/bash
# Этап 4: Проверка работоспособности развернутого сервиса

echo "[INFO] Проверка сервиса..."

curl -s -X POST http://127.0.0.1:5003/invocations \
    -H "Content-Type: application/json" \
    -d '{"dataframe_split": {"columns": ["Year", "Mileage", "Engine_size", "Brand_id", "Model_id", "Fuel_type_id"], "data": [[2019, 50000, 2.0, 1, 5, 1]]}}'

echo ""
echo "[SUCCESS] Сервис работает!"