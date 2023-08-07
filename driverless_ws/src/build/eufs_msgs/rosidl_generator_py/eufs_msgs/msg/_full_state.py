# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/FullState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_FullState(type):
    """Metaclass of message 'FullState'."""

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
                'eufs_msgs.msg.FullState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__full_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__full_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__full_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__full_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__full_state

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


class FullState(metaclass=Metaclass_FullState):
    """Message class 'FullState'."""

    __slots__ = [
        '_header',
        '_x_pos',
        '_y_pos',
        '_yaw',
        '_roll',
        '_u_x',
        '_u_y',
        '_yaw_mder',
        '_front_throttle',
        '_rear_throttle',
        '_steering',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'x_pos': 'double',
        'y_pos': 'double',
        'yaw': 'double',
        'roll': 'double',
        'u_x': 'double',
        'u_y': 'double',
        'yaw_mder': 'double',
        'front_throttle': 'double',
        'rear_throttle': 'double',
        'steering': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
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
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.x_pos = kwargs.get('x_pos', float())
        self.y_pos = kwargs.get('y_pos', float())
        self.yaw = kwargs.get('yaw', float())
        self.roll = kwargs.get('roll', float())
        self.u_x = kwargs.get('u_x', float())
        self.u_y = kwargs.get('u_y', float())
        self.yaw_mder = kwargs.get('yaw_mder', float())
        self.front_throttle = kwargs.get('front_throttle', float())
        self.rear_throttle = kwargs.get('rear_throttle', float())
        self.steering = kwargs.get('steering', float())

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
        if self.x_pos != other.x_pos:
            return False
        if self.y_pos != other.y_pos:
            return False
        if self.yaw != other.yaw:
            return False
        if self.roll != other.roll:
            return False
        if self.u_x != other.u_x:
            return False
        if self.u_y != other.u_y:
            return False
        if self.yaw_mder != other.yaw_mder:
            return False
        if self.front_throttle != other.front_throttle:
            return False
        if self.rear_throttle != other.rear_throttle:
            return False
        if self.steering != other.steering:
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
    def x_pos(self):
        """Message field 'x_pos'."""
        return self._x_pos

    @x_pos.setter
    def x_pos(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x_pos' field must be of type 'float'"
        self._x_pos = value

    @property
    def y_pos(self):
        """Message field 'y_pos'."""
        return self._y_pos

    @y_pos.setter
    def y_pos(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y_pos' field must be of type 'float'"
        self._y_pos = value

    @property
    def yaw(self):
        """Message field 'yaw'."""
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw' field must be of type 'float'"
        self._yaw = value

    @property
    def roll(self):
        """Message field 'roll'."""
        return self._roll

    @roll.setter
    def roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'roll' field must be of type 'float'"
        self._roll = value

    @property
    def u_x(self):
        """Message field 'u_x'."""
        return self._u_x

    @u_x.setter
    def u_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'u_x' field must be of type 'float'"
        self._u_x = value

    @property
    def u_y(self):
        """Message field 'u_y'."""
        return self._u_y

    @u_y.setter
    def u_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'u_y' field must be of type 'float'"
        self._u_y = value

    @property
    def yaw_mder(self):
        """Message field 'yaw_mder'."""
        return self._yaw_mder

    @yaw_mder.setter
    def yaw_mder(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw_mder' field must be of type 'float'"
        self._yaw_mder = value

    @property
    def front_throttle(self):
        """Message field 'front_throttle'."""
        return self._front_throttle

    @front_throttle.setter
    def front_throttle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'front_throttle' field must be of type 'float'"
        self._front_throttle = value

    @property
    def rear_throttle(self):
        """Message field 'rear_throttle'."""
        return self._rear_throttle

    @rear_throttle.setter
    def rear_throttle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rear_throttle' field must be of type 'float'"
        self._rear_throttle = value

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
