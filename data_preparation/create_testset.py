import os
import json

###
### Settings
###

MVTEC_BASE_PATH = "/home/luca_piai/big_disk/datasets/mvtec/"
OUTPUT_JSON = "test_set.json"

# Prompts used to test fine-tuned models.
PROMPTS = {
    "hazelnut": [
        "an hazelnut with a splitted apex",
        "add a crack to the surface of the hazelnut",
        "add dark rot spots to the hazelnut",
        "add burned surface spots to the hazelnut",
        "crushed hull",
        "add surface abrasions",
        "add surface mold",
        "add blue dots"

        #"add a deep scratch to the hazelnut",
        #"add a dark liquid stain to the hazelnut",
        #"add a sharp gash to the hazelnut",
        #"add a small puncture mark to the hazelnut",
    ],
    "pill": [
        "a pill with a chipped edge",
        "add a fracture to the surface of the pill",
        "a pill with a swollen, bloated shape",
        "a pill with an eroded surface",

        #"add a small hole to the pill",
        #"add a dirty smudge to the pill",
        #"add blue dots to the surface of the pill",
        #"a pill with a burned surface with black spots"
    ],
    "screw": [
        "a screw with a blunted tip",
        "a screw with flattened threads",
        "a screw with a cracked neck",
        "corrode the surface of the screw",
        "bent shank",
        "burred head rim",
        "gouged smooth shank",
        "peeling plating",

        #"a screw with a burred head rim",
        #"a screw with a bent shaft",
        #"add a scraped groove to the screw head",
        #"add a stripped area to the screw thread"
    ],
    "cable" : [
        "a cable with a melted insulation",
        "a cable with a cracked insulation",
        "a kinked cable",
        "a cable with a charred surface",
        "cut outer sheath",
        "a cable with rodent damage",
        "exposed conductor",
        "frayed strands"
        #"cables swap: two green one brown",
    ],
    "transistor" : [
        "add an exploded package to the transistor",
        "add bent or shorted leads to the transistor",
        "add a thermal burn mark to the transistor",
        "a transistor with corroded leads",
        "a transistor with a chipped edge",
        "cracked case",
        "chipped housing",
        "lead discoluration",
        "snapped terminal"
    ],
    "tile" : [
        "add a chemical stain to the tile",
        "add a chipped edge to the tile",
        "add a hairline crack to the tile",
        "add surface pitting to the tile"
    ],
    "leather" : [
        "add a puncture hole to the leather",
        "add mold or mildew to the leather",
        "add a liquid stain to the leather",
        "add a cigarette burn to the leather",
        "add a blue ink stain",
    ],
    "wood" : [
        "add hole",
        "add mold or mildew",
        "add a liquid stain",
        "add a cigarette burn",
        "add a hairline crack",
        "deep scratches",
        "burned black dots",
        "discoloration",
        "chemical stain",
        "rodent gnawing",
        "insect bore hole",
        "surface abrasions",
        "chipped edge"
    ],
    "carpet" : [
        "puncture hole",
        "sun fading",
        "add a cigarette burn",
        "deep scratches",
        "jagged tear",
        "discoloration",
        "chemical stain",
        "add a blue ink stain",
    ],
    "capsule" : [
        "defected printing",
        "add a crack",
        "a melted capsule",
        "add a small hole",
        "corroded surface",
        "chipped edge",
        "exploded capsule",
        "add mold",
        "add an embedded contaminant",
        "rusted surface"
    ], 
    # The categories below belongs to VISA!
    #"metal_nut": [
    #    "deep scratches on the surface",
    #    "burred rim",
    #    "peeling plating",
    #    "corroded surface",
    #    "hairline crack",
    #    "add a small hole",
    #    "add a chipped edge",
    #    "exploded metal nut",
    #    "add a liquid stain",
    #    "bloated shape",
    #    "rusted surface"
    #],
    #"candle": [
    #    "add a melted wax drip to the circles",
    #    "add a crack to each circle surface",
    #    "add a burn mark to the circles",
    #    "add a discoloration to the circles",
    #    "add a chip to the circles",
    #    "add a dent to the circles",
    #    "add a scratch to the circles",
    #    "add a stain to the circles"
    #],
}


def main():
    test_set = {}
    for category, prompts in PROMPTS.items():
        # Input images paths.
        paths = [
            os.path.join(MVTEC_BASE_PATH, category, "test", "good", f"{i:03d}.png") 
            for i in range(len(prompts))
        ]
        test_set[category] = {
            "inputs": paths,
            "prompts": prompts
        }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(test_set, f, indent=4)


if __name__ == '__main__':
    main()