import json

def main():
    hz = {
        "input": [f"hazelnut/test/good/{i:03d}.png" for i in range(10)],
        "prompt": [],
        "output": [],
    }
    pill = {
        "input": [f"pill/test/good/{i:03d}.png" for i in range(10)],
        "prompt": [],
        "output": []
    }

    out = {
        "hazelnut": hz,
        "pill": pill
    }

    with open("testset.json", "w") as f:
        json.dump(out, f, indent=4)


if __name__ == '__main__':
    main()