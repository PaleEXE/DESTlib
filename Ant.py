class Ant:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self._x = x
        self._y = y

    def move_up(self) -> None:
        self._y += 1

    def move_down(self) -> None:
        self._y -= 1

    def move_left(self) -> None:
        self._x -= 1

    def move_right(self) -> None:
        self._x += 1

    def __str__(self):
        return f"{self._x}, {self._y}"
