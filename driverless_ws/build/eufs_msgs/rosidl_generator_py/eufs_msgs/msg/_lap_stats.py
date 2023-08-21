# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/LapStats.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LapStats(type):
    """Metaclass of message 'LapStats'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('eufs_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'eufs_msgs.msg.LapStats')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__lap_stats
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__lap_stats
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__lap_stats
            cls._TYPE_SUPPORT = module.type_support_msg__msg__lap_stats
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__lap_stats

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LapStats(metaclass=Metaclass_LapStats):
    """Message class 'LapStats'."""

    __slots__ = [
        '_header',
        '_lap_number',
        '_lap_time',
        '_avg_speed',
        '_max_speed',
        '_speed_var',
        '_max_slip',
        '_max_lateral_accel',
        '_normalized_deviation_mse',
        '_deviation_var',
        '_max_deviation',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'lap_number': 'int64',
        'lap_time': 'double',
        'avg_speed': 'double',
        'max_speed': 'double',
        'speed_var': 'double',
        'max_slip': 'double',
        'max_lateral_accel': 'double',
        'normalized_deviation_mse': 'double',
        'deviation_var': 'double',
        'max_deviation': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.lap_number = kwargs.get('lap_number', int())
        self.lap_time = kwargs.get('lap_time', float())
        self.avg_speed = kwargs.get('avg_speed', float())
        self.max_speed = kwargs.get('max_speed', float())
        self.speed_var = kwargs.get('speed_var', float())
        self.max_slip = kwargs.get('max_slip', float())
        self.max_lateral_accel = kwargs.get('max_lateral_accel', float())
        self.normalized_deviation_mse = kwargs.get('normalized_deviation_mse', float())
        self.deviation_var = kwargs.get('deviation_var', float())
        self.max_deviation = kwargs.get('max_deviation', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.lap_number != other.lap_number:
            return False
        if self.lap_time != other.lap_time:
            return False
        if self.avg_speed != other.avg_speed:
            return False
        if self.max_speed != other.max_speed:
            return False
        if self.speed_var != other.speed_var:
            return False
        if self.max_slip != other.max_slip:
            return False
        if self.max_lateral_accel != other.max_lateral_accel:
            return False
        if self.normalized_deviation_mse != other.normalized_deviation_mse:
            return False
        if self.deviation_var != other.deviation_var:
            return False
        if self.max_deviation != other.max_deviation:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @property
    def lap_number(self):
        """Message field 'lap_number'."""
        return self._lap_number

    @lap_number.setter
    def lap_number(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'lap_number' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'lap_number' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._lap_number = value

    @property
    def lap_time(self):
        """Message field 'lap_time'."""
        return self._lap_time

    @lap_time.setter
    def lap_time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'lap_time' field must be of type 'float'"
        self._lap_time = value

    @property
    def avg_speed(self):
        """Message field 'avg_speed'."""
        return self._avg_speed

    @avg_speed.setter
    def avg_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'avg_speed' field must be of type 'float'"
        self._avg_speed = value

    @property
    def max_speed(self):
        """Message field 'max_speed'."""
        return self._max_speed

    @max_speed.setter
    def max_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_speed' field must be of type 'float'"
        self._max_speed = value

    @property
    def speed_var(self):
        """Message field 'speed_var'."""
        return self._speed_var

    @speed_var.setter
    def speed_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'speed_var' field must be of type 'float'"
        self._speed_var = value

    @property
    def max_slip(self):
        """Message field 'max_slip'."""
        return self._max_slip

    @max_slip.setter
    def max_slip(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_slip' field must be of type 'float'"
        self._max_slip = value

    @property
    def max_lateral_accel(self):
        """Message field 'max_lateral_accel'."""
        return self._max_lateral_accel

    @max_lateral_accel.setter
    def max_lateral_accel(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_lateral_accel' field must be of type 'float'"
        self._max_lateral_accel = value

    @property
    def normalized_deviation_mse(self):
        """Message field 'normalized_deviation_mse'."""
        return self._normalized_deviation_mse

    @normalized_deviation_mse.setter
    def normalized_deviation_mse(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'normalized_deviation_mse' field must be of type 'float'"
        self._normalized_deviation_mse = value

    @property
    def deviation_var(self):
        """Message field 'deviation_var'."""
        return self._deviation_var

    @deviation_var.setter
    def deviation_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'deviation_var' field must be of type 'float'"
        self._deviation_var = value

    @property
    def max_deviation(self):
        """Message field 'max_deviation'."""
        return self._max_deviation

    @max_deviation.setter
    def max_deviation(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_deviation' field must be of type 'float'"
        self._max_deviation = value
