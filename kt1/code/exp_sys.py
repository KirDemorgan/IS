import collections.abc
from experta import *

collections.Mapping = collections.abc.Mapping


class PlantFact(Fact):
    pass


class PlantExpertSystem(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.diagnoses = []
        self.fired_rules = []

    def reset(self):
        super().reset()
        self.diagnoses = []
        self.fired_rules = []

    @Rule(PlantFact(leaves_wilt='да'), PlantFact(soil_wet='да'), PlantFact(roots_dark='да'))
    def diagnose_root_rot(self):
        self.fired_rules.append("Правило 1: Диагностика корневой гнили")
        self.diagnoses.append(
            "Корневая гниль. Решение: Срочно извлечь растение, удалить загнившие корни, обработать фунгицидом.")

    @Rule(PlantFact(white_plaque='да'), PlantFact(high_humidity='да'), PlantFact(bad_ventilation='да'))
    def diagnose_powdery_mildew(self):
        self.fired_rules.append("Правило 2: Мучнистая роса")
        self.diagnoses.append(
            "Мучнистая роса. Решение: Удалить поражённые листья, обработать фунгицидом, снизить влажность.")

    @Rule(PlantFact(yellow_leaves='да'), PlantFact(green_veins='да'), PlantFact(in_shade='да'))
    def diagnose_chlorosis(self):
        self.fired_rules.append("Правило 3: Хлороз из-за нехватки света")
        self.diagnoses.append(
            "Хлороз. Решение: Переместить растение в освещённое место, внести железосодержащие удобрения.")

    @Rule(PlantFact(dry_spots='да'), PlantFact(direct_sun='да'))
    def diagnose_leaf_burn(self):
        self.fired_rules.append("Правило 4: Ожог листьев")
        self.diagnoses.append("Ожог листьев. Решение: Убрать из прямого солнца, обеспечить рассеянное освещение.")

    @Rule(PlantFact(web_on_leaves='да'), PlantFact(light_dots='да'), PlantFact(dry_air='да'))
    def diagnose_spider_mite(self):
        self.fired_rules.append("Правило 5: Поражение паутинным клещом")
        self.diagnoses.append(
            "Паутинный клещ. Решение: Изолировать растение, обработать акарицидом, повысить влажность.")

    @Rule(PlantFact(yellow_bottom_leaves='да'), PlantFact(soil_wet_long='да'), PlantFact(no_bad_smell='да'))
    def diagnose_overwatering(self):
        self.fired_rules.append("Правило 8: Переувлажнение без гнили")
        self.diagnoses.append("Переувлажнение. Решение: Сократить полив, улучшить дренаж.")


def ask_question(question_text):
    while True:
        answer = input(f"{question_text} (да/нет): ").strip().lower()
        if answer in ['да', 'нет']:
            return answer
        print("Ошибка ввода. Пожалуйста, введите 'да' или 'нет'.")


def run_interactive():
    engine = PlantExpertSystem()
    engine.reset()

    if ask_question("Листья вянут / желтеют снизу?") == 'да':
        engine.declare(PlantFact(leaves_wilt='да'))
        engine.declare(PlantFact(yellow_bottom_leaves='да'))
        if ask_question("Почва долго не просыхает / постоянно влажная?") == 'да':
            engine.declare(PlantFact(soil_wet='да'))
            engine.declare(PlantFact(soil_wet_long='да'))
            if ask_question("Наблюдается потемнение корней?") == 'да':
                engine.declare(PlantFact(roots_dark='да'))
            elif ask_question("Неприятный запах из горшка отсутствует?") == 'да':
                engine.declare(PlantFact(no_bad_smell='да'))

    if ask_question("Есть сухие светлые пятна на листьях?") == 'да':
        engine.declare(PlantFact(dry_spots='да'))
        if ask_question("Растение находится под прямыми солнечными лучами?") == 'да':
            engine.declare(PlantFact(direct_sun='да'))

    if ask_question("Есть тонкая паутина на листьях?") == 'да':
        engine.declare(PlantFact(web_on_leaves='да'))
        engine.declare(PlantFact(light_dots=ask_question("Есть мелкие светлые точки?")))
        engine.declare(PlantFact(dry_air=ask_question("Воздух в помещении сухой?")))

    engine.run()

    if engine.diagnoses:
        for d in engine.diagnoses:
            print(f"-> {d}")
        for rule in engine.fired_rules:
            print(f" - {rule}")
    else:
        print("Болезнь не распознана.")


def run_test_scenarios():
    print("--- Запуск сценариев верификации ---")
    engine = PlantExpertSystem()

    print("\nТест 1 (Use Case: Паутинный клещ):")
    engine.reset()
    engine.declare(PlantFact(web_on_leaves='да'), PlantFact(light_dots='да'), PlantFact(dry_air='да'))
    engine.run()
    for d in engine.diagnoses: print(f"-> {d}")

    print("\nТест 2 (Use Case: Ожог листьев):")
    engine.reset()
    engine.declare(PlantFact(dry_spots='да'), PlantFact(direct_sun='да'))
    engine.run()
    for d in engine.diagnoses: print(f"-> {d}")
    print("------------------------------------\n")


if __name__ == "__main__":
    run_test_scenarios()
    run_interactive()
