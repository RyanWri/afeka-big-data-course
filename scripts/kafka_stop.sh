#!/bin/bash

KAFKA_BIN_DIR="/usr/local/kafka/kafka_2.13-3.2.1/bin"
BOOTSTRAP_SERVER="localhost:9092"

echo "Stopping Kafka server"
$KAFKA_BIN_DIR/kafka-server-stop.sh
sleep 2
echo "Kafka server stopped"

echo "Stopping Zookeeper service"
$KAFKA_BIN_DIR/zookeeper-server-stop.sh
sleep 2
echo "Zookeeper service stopped"

