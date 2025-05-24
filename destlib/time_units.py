from typing import List, Dict, Union


class TimeUnit:
    Sec: str = "Seconds"
    Min: str = "Minutes"
    Hour: str = "Hours"
    Day: str = "Days"
    Week: str = "Weeks"
    Month: str = "Months"
    Year: str = "Years"

    _conversion_factors: Dict[str, int] = {
        Sec: 1,
        Min: 60,
        Hour: 3600,
        Day: 86400,
        Week: 604800,
        Month: 2592000,  # Assuming 30 days in a month
        Year: 31536000,  # Assuming 365 days in a year
    }

    @classmethod
    def all_units(cls) -> List[str]:
        """Returns all available time units."""
        return list(cls._conversion_factors.keys())

    @classmethod
    def is_valid_unit(cls, unit: str) -> bool:
        """Checks if a given unit is valid."""
        return unit in cls._conversion_factors

    @classmethod
    def to_seconds(cls, value: Union[int, float], unit: str) -> Union[int, float]:
        """
        Converts a given value and unit to seconds.

        Args:
            value (int | float): The value to convert.
            unit (str): The time unit.

        Returns:
            float: The equivalent value in seconds.

        Raises:
            ValueError: If the unit is invalid.
        """
        if not cls.is_valid_unit(unit):
            raise ValueError(f"Invalid time unit: {unit}")
        if value < 0:
            raise ValueError(f"Time value cannot be less than zero: {value}")
        return value * cls._conversion_factors[unit]

    @classmethod
    def from_seconds(
        cls, seconds: Union[int, float], target_unit: str
    ) -> Union[int, float]:
        """
        Converts seconds into the target time unit.

        Args:
            seconds (float): The number of seconds.
            target_unit (str): The desired time unit.

        Returns:
            float: The equivalent value in the target unit.

        Raises:
            ValueError: If the target unit is invalid.
        """
        if not cls.is_valid_unit(target_unit):
            raise ValueError(f"Invalid time unit: {target_unit}")
        if seconds < 0:
            raise ValueError(f"Time value cannot be less than zero: {seconds}")
        return seconds / cls._conversion_factors[target_unit]

    def __str__(self) -> str:
        return "TimeUnit class: A utility for handling time conversions and units."
