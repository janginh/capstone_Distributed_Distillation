#!/bin/bash
# [A5000] Kafka 토픽 초기화

set -e

echo "⏳ Kafka 연결 대기..."
until docker exec kafka1 kafka-topics.sh --bootstrap-server kafka1:9094 --list > /dev/null 2>&1; do
    sleep 2
done
echo "✅ Kafka 연결 완료"

echo ""
echo "📌 토픽 생성..."

docker exec kafka1 kafka-topics.sh --bootstrap-server kafka1:9094 \
    --create --topic raw-images --partitions 12 --replication-factor 2 \
    --if-not-exists --config max.message.bytes=20971520 --config retention.ms=259200000
echo "  ✅ raw-images (12p, repl=2, 20MB max, 72h)"

docker exec kafka1 kafka-topics.sh --bootstrap-server kafka1:9094 \
    --create --topic training-events --partitions 3 --replication-factor 2 \
    --if-not-exists
echo "  ✅ training-events (3p)"

echo ""
docker exec kafka1 kafka-topics.sh --bootstrap-server kafka1:9094 --list
echo ""
echo "🎉 완료!"
