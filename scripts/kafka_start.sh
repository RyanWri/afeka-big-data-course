#!/bin/bash

KAFKA_BIN_DIR="/usr/local/kafka/kafka_2.13-3.2.1/bin"
BOOTSTRAP_SERVER="localhost:9092"
DEFAULT_TOPIC="image-topic"

echo "Starting Zookeeper service"
$KAFKA_BIN_DIR/zookeeper-server-start.sh $KAFKA_BIN_DIR/../config/zookeeper.properties > /dev/null 2>&1 &
sleep 5
echo "Zookeeper service started"

echo "Starting Kafka server"
$KAFKA_BIN_DIR/kafka-server-start.sh $KAFKA_BIN_DIR/../config/server.properties > /dev/null 2>&1 &
sleep 5
echo "Kafka server started"

echo "Creating topic: $topic"
$KAFKA_BIN_DIR/kafka-topics.sh --create --topic "$DEFAULT_TOPIC" --bootstrap-server $BOOTSTRAP_SERVER --partitions 1 --replication-factor 1

if [ $? -eq 0 ]; then
    echo "Topic '$DEFAULT_TOPIC' created successfully!"
else
    echo "Failed to create topic '$DEFAULT_TOPIC'."
fi
