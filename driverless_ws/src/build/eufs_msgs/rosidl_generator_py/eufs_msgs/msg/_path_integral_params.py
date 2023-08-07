# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/PathIntegralParams.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PathIntegralParams(type):
    """Metaclass of message 'PathIntegralParams'."""

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
                'eufs_msgs.msg.PathIntegralParams')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__path_integral_params
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__path_integral_params
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__path_integral_params
            cls._TYPE_SUPPORT = module.type_support_msg__msg__path_integral_params
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__path_integral_params

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PathIntegralParams(metaclass=Metaclass_PathIntegralParams):
    """Message class 'PathIntegralParams'."""

    __slots__ = [
        '_hz',
        '_num_timesteps',
        '_num_iters',
        '_gamma',
        '_init_steering',
        '_init_throttle',
        '_steering_var',
        '_throttle_var',
        '_max_throttle',
        '_speed_coefficient',
        '_track_coefficient',
        '_max_slip_angle',
        '_track_slop',
        '_crash_coeff',
        '_map_path',
        '_desired_speed',
    ]

    _fields_and_field_types = {
        'hz': 'int64',
        'num_timesteps': 'int64',
        'num_iters': 'int64',
        'gamma': 'double',
        'init_steering': 'double',
        'init_throttle': 'double',
        'steering_var': 'double',
        'throttle_var': 'double',
        'max_throttle': 'double',
        'speed_coefficient': 'double',
        'track_coefficient': 'double',
        'max_slip_angle': 'double',
        'track_slop': 'double',
        'crash_coeff': 'double',
        'map_path': 'string',
        'desired_speed': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
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
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.hz = kwargs.get('hz', int())
        self.num_timesteps = kwargs.get('num_timesteps', int())
        self.num_iters = kwargs.get('num_iters', int())
        self.gamma = kwargs.get('gamma', float())
        self.init_steering = kwargs.get('init_steering', float())
        self.init_throttle = kwargs.get('init_throttle', float())
        self.steering_var = kwargs.get('steering_var', float())
        self.throttle_var = kwargs.get('throttle_var', float())
        self.max_throttle = kwargs.get('max_throttle', float())
        self.speed_coefficient = kwargs.get('speed_coefficient', float())
        self.track_coefficient = kwargs.get('track_coefficient', float())
        self.max_slip_angle = kwargs.get('max_slip_angle', float())
        self.track_slop = kwargs.get('track_slop', float())
        self.crash_coeff = kwargs.get('crash_coeff', float())
        self.map_path = kwargs.get('map_path', str())
        self.desired_speed = kwargs.get('desired_speed', float())

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
        if self.hz != other.hz:
            return False
        if self.num_timesteps != other.num_timesteps:
            return False
        if self.num_iters != other.num_iters:
            return False
        if self.gamma != other.gamma:
            return False
        if self.init_steering != other.init_steering:
            return False
        if self.init_throttle != other.init_throttle:
            return False
        if self.steering_var != other.steering_var:
            return False
        if self.throttle_var != other.throttle_var:
            return False
        if self.max_throttle != other.max_throttle:
            return False
        if self.speed_coefficient != other.speed_coefficient:
            return False
        if self.track_coefficient != other.track_coefficient:
            return False
        if self.max_slip_angle != other.max_slip_angle:
            return False
        if self.track_slop != other.track_slop:
            return False
        if self.crash_coeff != other.crash_coeff:
            return False
        if self.map_path != other.map_path:
            return False
        if self.desired_speed != other.desired_speed:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def hz(self):
        """Message field 'hz'."""
        return self._hz

    @hz.setter
    def hz(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'hz' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'hz' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._hz = value

    @property
    def num_timesteps(self):
        """Message field 'num_timesteps'."""
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_timesteps' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'num_timesteps' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._num_timesteps = value

    @property
    def num_iters(self):
        """Message field 'num_iters'."""
        return self._num_iters

    @num_iters.setter
    def num_iters(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_iters' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'num_iters' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._num_iters = value

    @property
    def gamma(self):
        """Message field 'gamma'."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'gamma' field must be of type 'float'"
        self._gamma = value

    @property
    def init_steering(self):
        """Message field 'init_steering'."""
        return self._init_steering

    @init_steering.setter
    def init_steering(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'init_steering' field must be of type 'float'"
        self._init_steering = value

    @property
    def init_throttle(self):
        """Message field 'init_throttle'."""
        return self._init_throttle

    @init_throttle.setter
    def init_throttle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'init_throttle' field must be of type 'float'"
        self._init_throttle = value

    @property
    def steering_var(self):
        """Message field 'steering_var'."""
        return self._steering_var

    @steering_var.setter
    def steering_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'steering_var' field must be of type 'float'"
        self._steering_var = value

    @property
    def throttle_var(self):
        """Message field 'throttle_var'."""
        return self._throttle_var

    @throttle_var.setter
    def throttle_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'throttle_var' field must be of type 'float'"
        self._throttle_var = value

    @property
    def max_throttle(self):
        """Message field 'max_throttle'."""
        return self._max_throttle

    @max_throttle.setter
    def max_throttle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_throttle' field must be of type 'float'"
        self._max_throttle = value

    @property
    def speed_coefficient(self):
        """Message field 'speed_coefficient'."""
        return self._speed_coefficient

    @speed_coefficient.setter
    def speed_coefficient(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'speed_coefficient' field must be of type 'float'"
        self._speed_coefficient = value

    @property
    def track_coefficient(self):
        """Message field 'track_coefficient'."""
        return self._track_coefficient

    @track_coefficient.setter
    def track_coefficient(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'track_coefficient' field must be of type 'float'"
        self._track_coefficient = value

    @property
    def max_slip_angle(self):
        """Message field 'max_slip_angle'."""
        return self._max_slip_angle

    @max_slip_angle.setter
    def max_slip_angle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_slip_angle' field must be of type 'float'"
        self._max_slip_angle = value

    @property
    def track_slop(self):
        """Message field 'track_slop'."""
        return self._track_slop

    @track_slop.setter
    def track_slop(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'track_slop' field must be of type 'float'"
        self._track_slop = value

    @property
    def crash_coeff(self):
        """Message field 'crash_coeff'."""
        return self._crash_coeff

    @crash_coeff.setter
    def crash_coeff(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'crash_coeff' field must be of type 'float'"
        self._crash_coeff = value

    @property
    def map_path(self):
        """Message field 'map_path'."""
        return self._map_path

    @map_path.setter
    def map_path(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'map_path' field must be of type 'str'"
        self._map_path = value

    @property
    def desired_speed(self):
        """Message field 'desired_speed'."""
        return self._desired_speed

    @desired_speed.setter
    def desired_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'desired_speed' field must be of type 'float'"
        self._desired_speed = value
