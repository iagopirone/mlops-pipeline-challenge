import json


message_as_string = """
{
  "event": "data.build",
  "trigger": "manual",
  "raw_uri": "data/raw",
  "params": {
    "val_frac": 0.15,
    "test_frac": 0.15,
    "seed": 42
  }
}
"""


def main() -> None:
    print("Mensagem como string:")
    print(message_as_string)

    message_as_dict = json.loads(message_as_string)

    print("\nMensagem convertida para dicionário Python:")
    print(message_as_dict)

    print("\nAcessando campos específicos:")
    print("event:", message_as_dict["event"])
    print("raw_uri:", message_as_dict["raw_uri"])
    print("val_frac:", message_as_dict["params"]["val_frac"])
    print("test_frac:", message_as_dict["params"]["test_frac"])
    print("seed:", message_as_dict["params"]["seed"])


if __name__ == "__main__":
    main()
