# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/ChassisState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ChassisState(type):
    """Metaclass of message 'ChassisState'."""

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
                'eufs_msgs.msg.ChassisState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__chassis_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__chassis_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__chassis_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__chassis_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__chassis_state

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


class ChassisState(metaclass=Metaclass_ChassisState):
    """Message class 'ChassisState'."""

    __slots__ = [
        '_header',
        '_throttle_relay_enabled',
        '_autonomous_enabled',
        '_runstop_motion_enabled',
        '_steering_commander',
        '_steering',
        '_throttle_commander',
        '_throttle',
        '_front_brake_commander',
        '_front_brake',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'throttle_relay_enabled': 'boolean',
        'autonomous_enabled': 'boolean',
        'runstop_motion_enabled': 'boolean',
        'steering_commander': 'string',
        'steering': 'double',
        'throttle_commander': 'string',
        'throttle': 'double',
        'front_brake_commander': 'string',
        'front_brake': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.throttle_relay_enabled = kwargs.get('throttle_relay_enabled', bool())
        self.autonomous_enabled = kwargs.get('autonomous_enabled', bool())
        self.runstop_motion_enabled = kwargs.get('runstop_motion_enabled', bool())
        self.steering_commander = kwargs.get('steering_commander', str())
        self.steering = kwargs.get('steering', float())
        self.throttle_commander = kwargs.get('throttle_commander', str())
        self.throttle = kwargs.get('throttle', float())
        self.front_brake_commander = kwargs.get('front_brake_commander', str())
        self.front_brake = kwargs.get('front_brake', float())

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
        if self.throttle_relay_enabled != other.throttle_relay_enabled:
            return False
        if self.autonomous_enabled != other.autonomous_enabled:
            return False
        if self.runstop_motion_enabled != other.runstop_motion_enabled:
            return False
        if self.steering_commander != other.steering_commander:
            return False
        if self.steering != other.steering:
            return False
        if self.throttle_commander != other.throttle_commander:
            return False
        if self.throttle != other.throttle:
            return False
        if self.front_brake_commander != other.front_brake_commander:
            return False
        if self.front_brake != other.front_brake:
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
    def throttle_relay_enabled(self):
        """Message field 'throttle_relay_enabled'."""
        return self._throttle_relay_enabled

    @throttle_relay_enabled.setter
    def throttle_relay_enabled(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'throttle_relay_enabled' field must be of type 'bool'"
        self._throttle_relay_enabled = value

    @property
    def autonomous_enabled(self):
        """Message field 'autonomous_enabled'."""
        return self._autonomous_enabled

    @autonomous_enabled.setter
    def autonomous_enabled(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'autonomous_enabled' field must be of type 'bool'"
        self._autonomous_enabled = value

    @property
    def runstop_motion_enabled(self):
        """Message field 'runstop_motion_enabled'."""
        return self._runstop_motion_enabled

    @runstop_motion_enabled.setter
    def runstop_motion_enabled(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'runstop_motion_enabled' field must be of type 'bool'"
        self._runstop_motion_enabled = value

    @property
    def steering_commander(self):
        """Message field 'steering_commander'."""
        return self._steering_commander

    @steering_commander.setter
    def steering_commander(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'steering_commander' field must be of type 'str'"
        self._steering_commander = value

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
    def throttle_commander(self):
        """Message field 'throttle_commander'."""
        return self._throttle_commander

    @throttle_commander.setter
    def throttle_commander(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'throttle_commander' field must be of type 'str'"
        self._throttle_commander = value

    @property
    def throttle(self):
        """Message field 'throttle'."""
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'throttle' field must be of type 'float'"
        self._throttle = value

    @property
    def front_brake_commander(self):
        """Message field 'front_brake_commander'."""
        return self._front_brake_commander

    @front_brake_commander.setter
    def front_brake_commander(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'front_brake_commander' field must be of type 'str'"
        self._front_brake_commander = value

    @property
    def front_brake(self):
        """Message field 'front_brake'."""
        return self._front_brake

    @front_brake.setter
    def front_brake(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'front_brake' field must be of type 'float'"
        self._front_brake = value
