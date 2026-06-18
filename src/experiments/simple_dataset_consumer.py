import json

import pika


QUEUE_NAME = "q.example.dataset"


def on_message(channel, method, properties, body) -> None:
    print("\n[consumer] mensagem recebida como bytes:")
    print(body)

    message_as_string = body.decode("utf-8")

    print("\n[consumer] mensagem convertida para string:")
    print(message_as_string)

    message = json.loads(message_as_string)

    print("\n[consumer] mensagem convertida para dicionário:")
    print(message)

    raw_uri = message["raw_uri"]
    val_frac = message["params"]["val_frac"]
    test_frac = message["params"]["test_frac"]
    seed = message["params"]["seed"]

    print("\n[consumer] parâmetros extraídos da mensagem:")
    print("raw_uri:", raw_uri)
    print("val_frac:", val_frac)
    print("test_frac:", test_frac)
    print("seed:", seed)

    print("\n[consumer] aqui eu chamaria a lógica do prep_data.py usando esses parâmetros.")

    channel.basic_ack(delivery_tag=method.delivery_tag)

    print("\n[consumer] ack enviado. Mensagem processada com sucesso.")

    channel.stop_consuming()


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=QUEUE_NAME,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print("[consumer] aguardando mensagem na fila:", QUEUE_NAME)

    channel.start_consuming()

    connection.close()


if __name__ == "__main__":
    main()
