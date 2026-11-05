"""Écrit un petit jeu de recettes rédigées à la main.

Il sert aux tests et au mode démo hors ligne. Il ne remplace pas le vrai dataset
créé par generate_dataset.py : 12 exemples ne suffisent pas pour un fine-tuning.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "shared"))

from recette_core.contract import Recipe, build_user_prompt, render_recipe  # noqa: E402

SAMPLES = [
    (["œufs", "courgettes", "feta", "citron"], ["plaques"], 20, Recipe(
        "Omelette verte à la feta citronnée", 15, ["plaques"],
        ["œufs : 3", "courgettes : 1", "feta : 60 g", "citron : 1/2", "huile d'olive : 1 c. à soupe"],
        ["Râpe la courgette et fais-la dorer 4 minutes à l'huile.", "Bats les œufs avec le zeste de citron.", "Verse sur la courgette et laisse prendre 3 minutes.", "Émiette la feta et plie l'omelette."])),
    (["pois chiches", "tomates", "curry", "lait de coco"], ["plaques"], 25, Recipe(
        "Curry express tomate et coco", 20, ["plaques"],
        ["pois chiches : 240 g", "tomates : 3", "curry : 2 c. à café", "lait de coco : 200 ml", "sel : 1 pincée"],
        ["Coupe les tomates en dés.", "Fais revenir le curry 1 minute à sec.", "Ajoute les tomates et laisse compoter 6 minutes.", "Ajoute les pois chiches et le lait de coco, puis mijote 10 minutes."])),
    (["blanc de poulet", "pommes de terre", "paprika", "ail"], ["four"], 60, Recipe(
        "Poulet fumé et pommes de terre en plaque", 50, ["four"],
        ["blanc de poulet : 300 g", "pommes de terre : 500 g", "paprika : 1 c. à soupe", "ail : 2 gousses", "huile : 2 c. à soupe"],
        ["Préchauffe le four à 200 °C.", "Coupe les pommes de terre en quartiers et mélange-les à l'huile et au paprika.", "Enfourne 25 minutes.", "Ajoute le poulet et l'ail, puis cuis 20 minutes jusqu'à cuisson à cœur."])),
    (["saumon", "riz", "sauce soja", "gingembre"], ["micro-ondes"], 15, Recipe(
        "Bol de saumon vapeur au gingembre", 12, ["micro-ondes"],
        ["saumon : 150 g", "riz : 80 g", "sauce soja : 2 c. à soupe", "gingembre : 1 cm", "eau : 160 ml"],
        ["Cuis le riz avec l'eau 8 minutes dans un bol couvert.", "Pose le saumon sur le riz avec le gingembre et la sauce soja.", "Cuis 3 minutes à puissance moyenne jusqu'à ce qu'il soit opaque."])),
    (["patate douce", "feta", "cumin"], ["airfryer"], 30, Recipe(
        "Frites de patate douce au cumin et feta", 25, ["airfryer"],
        ["patate douce : 2", "feta : 80 g", "cumin : 1 c. à café", "huile : 1 c. à soupe"],
        ["Coupe la patate douce en bâtonnets.", "Mélange avec l'huile et le cumin.", "Cuis 20 minutes à 190 °C en secouant à mi-cuisson.", "Émiette la feta avant de servir."])),
    (["carottes", "lait de coco", "curry", "oignon"], ["plaques", "mixeur"], 30, Recipe(
        "Velouté de carottes coco-curry", 28, ["plaques", "mixeur"],
        ["carottes : 500 g", "lait de coco : 200 ml", "curry : 1 c. à café", "oignon : 1", "eau : 500 ml"],
        ["Émince l'oignon et fais-le suer.", "Ajoute les carottes en rondelles, le curry et l'eau.", "Laisse cuire 20 minutes.", "Mixe avec le lait de coco."])),
    (["pâtes", "champignons", "crème fraîche", "persil"], ["plaques"], 20, Recipe(
        "Pâtes crémeuses aux champignons poêlés", 18, ["plaques"],
        ["pâtes : 200 g", "champignons : 250 g", "crème fraîche : 100 g", "persil : 2 branches", "sel : 1 pincée"],
        ["Fais cuire les pâtes dans l'eau salée.", "Poêle les champignons 6 minutes à feu vif.", "Ajoute la crème et une louche d'eau de cuisson.", "Mélange aux pâtes et parsème de persil."])),
    (["pomme", "pain rassis", "miel"], ["four"], 40, Recipe(
        "Pudding rustique pomme et miel", 35, ["four"],
        ["pomme : 2", "pain rassis : 150 g", "miel : 2 c. à soupe", "eau : 50 ml"],
        ["Préchauffe le four à 180 °C.", "Coupe le pain et les pommes en dés.", "Mélange avec le miel et l'eau dans un plat.", "Enfourne 25 minutes jusqu'à ce que le dessus dore."])),
    (["lentilles", "carottes", "oignon", "moutarde"], ["plaques", "autocuiseur"], 40, Recipe(
        "Lentilles fondantes à la moutarde", 30, ["autocuiseur"],
        ["lentilles : 250 g", "carottes : 2", "oignon : 1", "moutarde : 1 c. à soupe", "eau : 750 ml"],
        ["Coupe les carottes et l'oignon.", "Mets tout sauf la moutarde dans l'autocuiseur.", "Cuis 15 minutes sous pression.", "Ajoute la moutarde au moment de servir."])),
    (["tortillas", "haricots rouges", "maïs", "gruyère"], ["four"], 25, Recipe(
        "Tortillas gratinées haricots et maïs", 20, ["four"],
        ["tortillas : 4", "haricots rouges : 200 g", "maïs : 100 g", "gruyère : 80 g"],
        ["Préchauffe le four à 200 °C.", "Écrase légèrement les haricots avec le maïs.", "Garnis les tortillas puis roule-les.", "Couvre de gruyère et enfourne 10 minutes."])),
    (["tofu", "brocoli", "sauce soja", "miel"], ["plaques"], 20, Recipe(
        "Tofu laqué au miel et brocoli croquant", 18, ["plaques"],
        ["tofu : 200 g", "brocoli : 300 g", "sauce soja : 3 c. à soupe", "miel : 1 c. à soupe", "huile : 1 c. à soupe"],
        ["Détaille le brocoli et fais-le cuire 4 minutes à l'eau bouillante.", "Fais dorer le tofu en cubes 6 minutes.", "Ajoute le miel et la sauce soja, puis laisse laquer 2 minutes.", "Mélange avec le brocoli."])),
    (["banane", "yaourt nature", "fruits rouges surgelés"], ["mixeur"], 10, Recipe(
        "Smoothie bowl banane et fruits rouges", 5, ["mixeur"],
        ["banane : 2", "yaourt nature : 1", "fruits rouges surgelés : 150 g"],
        ["Coupe les bananes en morceaux.", "Mixe avec le yaourt et les fruits rouges.", "Sers aussitôt dans un bol."])),
]

SAMPLES_EN = [
    (["eggs", "kimchi", "cooked rice"], ["stove"], 15, "Korean", Recipe(
        "Crispy Kimchi Rice with a Jammy Egg", 12, ["stove"],
        ["cooked rice: 200 g", "kimchi: 100 g", "eggs: 2", "oil: 1 tbsp"],
        ["Fry the kimchi in the oil for 2 minutes.", "Add the rice and press it flat until crisp.", "Fry the eggs until the whites are fully set.", "Serve the eggs over the rice."])),
    (["chickpeas", "tomatoes", "harissa", "lemon"], ["stove"], 20, "Moroccan", Recipe(
        "Smoky Harissa Chickpea Skillet", 18, ["stove"],
        ["chickpeas: 240 g", "tomatoes: 3", "harissa: 1 tbsp", "lemon: 1/2", "olive oil: 1 tbsp"],
        ["Chop the tomatoes.", "Toast the harissa in the oil for 1 minute.", "Add the tomatoes and simmer for 6 minutes.", "Stir in the chickpeas, heat through and finish with lemon."])),
    (["sweet potato", "peanut butter", "coconut milk", "curry powder"], ["stove", "blender"], 30, "Senegalese", Recipe(
        "Silky Sweet Potato and Peanut Soup", 28, ["stove", "blender"],
        ["sweet potato: 500 g", "peanut butter: 2 tbsp", "coconut milk: 200 ml", "curry powder: 1 tsp", "water: 400 ml"],
        ["Dice the sweet potato.", "Simmer it with the water and curry powder for 18 minutes.", "Add the peanut butter and coconut milk.", "Blend until smooth."])),
]


def main() -> None:
    target = ROOT / "data" / "sample_recipes.jsonl"
    target.parent.mkdir(exist_ok=True)
    with target.open("w", encoding="utf-8") as sink:
        entries = [("fr", *sample[:3], "surprise", sample[3]) for sample in SAMPLES]
        entries += [("en", *sample) for sample in SAMPLES_EN]
        for language, ingredients, appliances, minutes, cuisine, recipe in entries:
            row = {
                "language": language,
                "cuisine": cuisine,
                "ingredients": ingredients,
                "appliances": appliances,
                "max_minutes": minutes,
                "user_prompt": build_user_prompt(
                    ingredients, appliances, minutes, language, cuisine
                ),
                "recipe": render_recipe(recipe, language),
            }
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(entries)} recettes écrites dans {target}")


if __name__ == "__main__":
    main()
