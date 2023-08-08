# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/WheelSpeeds.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_WheelSpeeds(type):
    """Metaclass of message 'WheelSpeeds'."""

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
                'eufs_msgs.msg.WheelSpeeds')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__wheel_speeds
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__wheel_speeds
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__wheel_speeds
            cls._TYPE_SUPPORT = module.type_support_msg__msg__wheel_speeds
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__wheel_speeds

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class WheelSpeeds(metaclass=Metaclass_WheelSpeeds):
    """Message class 'WheelSpeeds'."""

    __slots__ = [
        '_steering',
        '_lf_speed',
        '_rf_speed',
        '_lb_speed',
        '_rb_speed',
    ]

    _fields_and_field_types = {
        'steering': 'float',
        'lf_speed': 'float',
        'rf_speed': 'float',
        'lb_speed': 'float',
        'rb_speed': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.steering = kwargs.get('steering', float())
        self.lf_speed = kwargs.get('lf_speed', float())
        self.rf_speed = kwargs.get('rf_speed', float())
        self.lb_speed = kwargs.get('lb_speed', float())
        self.rb_speed = kwargs.get('rb_speed', float())

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
        if self.steering != other.steering:
            return False
        if self.lf_speed != other.lf_speed:
            return False
        if self.rf_speed != other.rf_speed:
            return False
        if self.lb_speed != other.lb_speed:
            return False
        if self.rb_speed != other.rb_speed:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def steering(self):
        """Message field 'steering'."""
        return self._steering

    @steering.setter
    def steering(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'steering' field must be of type 'float'"
        self._steering = value

    @property
    def lf_speed(self):
        """Message field 'lf_speed'."""
        return self._lf_speed

    @lf_speed.setter
    def lf_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'lf_speed' field must be of type 'float'"
        self._lf_speed = value

    @property
    def rf_speed(self):
        """Message field 'rf_speed'."""
        return self._rf_speed

    @rf_speed.setter
    def rf_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rf_speed' field must be of type 'float'"
        self._rf_speed = value

    @property
    def lb_speed(self):
        """Message field 'lb_speed'."""
        return self._lb_speed

    @lb_speed.setter
    def lb_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'lb_speed' field must be of type 'float'"
        self._lb_speed = value

    @property
    def rb_speed(self):
        """Message field 'rb_speed'."""
        return self._rb_speed

    @rb_speed.setter
    def rb_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rb_speed' field must be of type 'float'"
        self._rb_speed = value
