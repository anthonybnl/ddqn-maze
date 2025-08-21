import json
import random


class Labyrinthe:
    def __init__(self, size: int):
        if size < 2:
            raise Exception("lab size must be >= 2")

        self.LAB_SIZE = size
        self.state_len = size * size

        self.state = [i for i in range(self.state_len)]
        self.transition: list[list[int]] = []
        self.murs: list[tuple[int, int]] = []

        # self.generate()

    def generate(self):

        self.transition = [([]) for _ in range(self.state_len)]
        self.murs = []

        for i in range(self.LAB_SIZE):
            for j in range(self.LAB_SIZE - 1):
                ix = i * self.LAB_SIZE + j
                self.murs.append((ix, ix + 1))

        for i in range(self.LAB_SIZE - 1):
            for j in range(self.LAB_SIZE):
                ix = i * self.LAB_SIZE + j
                self.murs.append((ix, ix + self.LAB_SIZE))

        self.idx_cell = [i for i in range(self.state_len)]  # pour chaque cell, son id
        self.reverse_idx_cell = [
            [i] for i in range(self.state_len)
        ]  # pour chaque id, la liste des cell

        # on ouvre un des murs de la fin
        end_state_walls_ix = [
            i
            for i in range(len(self.murs))
            if (self.murs[i])[1] == (self.LAB_SIZE * self.LAB_SIZE - 1)
        ]

        self.open_mur_by_ix(
            end_state_walls_ix[random.randrange(len(end_state_walls_ix))]
        )

        for i in range(self.LAB_SIZE * self.LAB_SIZE - 2):
            self.open_mur_randomly_except_end_state()

    def open_mur_by_ix(self, ix: int):
        u, v = self.murs[ix]
        id_cell_gagnante = self.idx_cell[u]
        id_cell_perdante = self.idx_cell[v]

        if id_cell_gagnante == id_cell_perdante:
            raise Exception("same id_cell")

        self.murs.pop(ix)
        self.transition[u].append(v)
        self.transition[v].append(u)

        for cell in self.reverse_idx_cell[id_cell_perdante]:
            self.idx_cell[cell] = id_cell_gagnante
            self.reverse_idx_cell[id_cell_gagnante].append(cell)

        self.reverse_idx_cell[id_cell_perdante] = []

    def open_mur_randomly_except_end_state(self):

        trouve = False
        ix = -1

        while not trouve:
            ix = random.randrange(len(self.murs))
            u, v = self.murs[ix]
            id_cell_gagnante = self.idx_cell[u]
            id_cell_perdante = self.idx_cell[v]

            if (v != (self.LAB_SIZE * self.LAB_SIZE - 1)) and (
                id_cell_gagnante != id_cell_perdante
            ):
                trouve = True

        self.open_mur_by_ix(ix)

    def export_json(self):
        return json.dumps(
            {"size": self.LAB_SIZE, "transitions": self.transition, "murs": self.murs}
        )

    @staticmethod
    def load_from_json(json_str: str):
        obj = json.loads(json_str)
        lab = Labyrinthe(obj["size"])

        lab.transition = obj["transitions"]
        lab.murs = obj["murs"]

        return lab
