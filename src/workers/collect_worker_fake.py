import json
from datetime import datetime, timezone

import pika


QUEUES = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
]


LOW_CONF_THRESHOLD = 0.50


def declare_queues(channel) -> None:
    for queue in QUEUES:
        channel.queue_declare(queue=queue, durable=True)


def on_message(channel, method, properties, body) -> None:
    print("\n[collect_worker] mensagem recebida de q.infer.result:")
    print(body.decode("utf-8"))

    message = json.loads(body)

    min_conf = message.get("min_conf", 1.0)
    should_collect = min_conf < LOW_CONF_THRESHOLD

    if should_collect:
        label_task_event = {
            "event": "label.task",
            "inference_id": message["inference_id"],
            "image_uri": message["image_uri"],
            "reason": "low_confidence",
            "min_conf": min_conf,
            "threshold": LOW_CONF_THRESHOLD,
            "annotation_source": "oracle_simulado",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        data_build_event = {
            "event": "data.build",
            "trigger": "feedback",
            "raw_uri": "data/raw",
            "params": {
                "val_frac": 0.15,
                "test_frac": 0.15
            },
            "source_inference_id": message["inference_id"],
            "reason": "new_labeled_sample_from_feedback",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        channel.basic_publish(
            exchange="",
            routing_key="q.label.task",
            body=json.dumps(label_task_event).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )

        channel.basic_publish(
            exchange="",
            routing_key="q.data.build",
            body=json.dumps(data_build_event).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )

        print("[collect_worker] caso selecionado para anotação por baixa confiança.")
        print("[collect_worker] mensagem publicada em q.label.task:")
        print(json.dumps(label_task_event, indent=2))
        print("[collect_worker] mensagem publicada em q.data.build:")
        print(json.dumps(data_build_event, indent=2))

    else:
        print("[collect_worker] caso não selecionado para coleta. Confiança suficiente.")

    channel.basic_ack(delivery_tag=method.delivery_tag)


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    declare_queues(channel)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="q.infer.result",
        on_message_callback=on_message,
        auto_ack=False,
    )

    print("[collect_worker] aguardando mensagens em q.infer.result...")
    print("[collect_worker] pressione CTRL+C para parar.")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[collect_worker] encerrando...")
        channel.stop_consuming()

    connection.close()


if __name__ == "__main__":
    main()
