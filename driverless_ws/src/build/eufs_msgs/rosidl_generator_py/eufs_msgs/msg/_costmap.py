# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/Costmap.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'channel0'
# Member 'channel1'
# Member 'channel2'
# Member 'channel3'
import array  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Costmap(type):
    """Metaclass of message 'Costmap'."""

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
                'eufs_msgs.msg.Costmap')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__costmap
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__costmap
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__costmap
            cls._TYPE_SUPPORT = module.type_support_msg__msg__costmap
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__costmap

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Costmap(metaclass=Metaclass_Costmap):
    """Message class 'Costmap'."""

    __slots__ = [
        '_x_bounds_min',
        '_x_bounds_max',
        '_y_bounds_min',
        '_y_bounds_max',
        '_pixels_per_meter',
        '_channel0',
        '_channel1',
        '_channel2',
        '_channel3',
    ]

    _fields_and_field_types = {
        'x_bounds_min': 'double',
        'x_bounds_max': 'double',
        'y_bounds_min': 'double',
        'y_bounds_max': 'double',
        'pixels_per_meter': 'double',
        'channel0': 'sequence<float>',
        'channel1': 'sequence<float>',
        'channel2': 'sequence<float>',
        'channel3': 'sequence<float>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.x_bounds_min = kwargs.get('x_bounds_min', float())
        self.x_bounds_max = kwargs.get('x_bounds_max', float())
        self.y_bounds_min = kwargs.get('y_bounds_min', float())
        self.y_bounds_max = kwargs.get('y_bounds_max', float())
        self.pixels_per_meter = kwargs.get('pixels_per_meter', float())
        self.channel0 = array.array('f', kwargs.get('channel0', []))
        self.channel1 = array.array('f', kwargs.get('channel1', []))
        self.channel2 = array.array('f', kwargs.get('channel2', []))
        self.channel3 = array.array('f', kwargs.get('channel3', []))

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
        if self.x_bounds_min != other.x_bounds_min:
            return False
        if self.x_bounds_max != other.x_bounds_max:
            return False
        if self.y_bounds_min != other.y_bounds_min:
            return False
        if self.y_bounds_max != other.y_bounds_max:
            return False
        if self.pixels_per_meter != other.pixels_per_meter:
            return False
        if self.channel0 != other.channel0:
            return False
        if self.channel1 != other.channel1:
            return False
        if self.channel2 != other.channel2:
            return False
        if self.channel3 != other.channel3:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def x_bounds_min(self):
        """Message field 'x_bounds_min'."""
        return self._x_bounds_min

    @x_bounds_min.setter
    def x_bounds_min(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x_bounds_min' field must be of type 'float'"
        self._x_bounds_min = value

    @property
    def x_bounds_max(self):
        """Message field 'x_bounds_max'."""
        return self._x_bounds_max

    @x_bounds_max.setter
    def x_bounds_max(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x_bounds_max' field must be of type 'float'"
        self._x_bounds_max = value

    @property
    def y_bounds_min(self):
        """Message field 'y_bounds_min'."""
        return self._y_bounds_min

    @y_bounds_min.setter
    def y_bounds_min(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y_bounds_min' field must be of type 'float'"
        self._y_bounds_min = value

    @property
    def y_bounds_max(self):
        """Message field 'y_bounds_max'."""
        return self._y_bounds_max

    @y_bounds_max.setter
    def y_bounds_max(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y_bounds_max' field must be of type 'float'"
        self._y_bounds_max = value

    @property
    def pixels_per_meter(self):
        """Message field 'pixels_per_meter'."""
        return self._pixels_per_meter

    @pixels_per_meter.setter
    def pixels_per_meter(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pixels_per_meter' field must be of type 'float'"
        self._pixels_per_meter = value

    @property
    def channel0(self):
        """Message field 'channel0'."""
        return self._channel0

    @channel0.setter
    def channel0(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'channel0' array.array() must have the type code of 'f'"
            self._channel0 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'channel0' field must be a set or sequence and each value of type 'float'"
        self._channel0 = array.array('f', value)

    @property
    def channel1(self):
        """Message field 'channel1'."""
        return self._channel1

    @channel1.setter
    def channel1(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'channel1' array.array() must have the type code of 'f'"
            self._channel1 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'channel1' field must be a set or sequence and each value of type 'float'"
        self._channel1 = array.array('f', value)

    @property
    def channel2(self):
        """Message field 'channel2'."""
        return self._channel2

    @channel2.setter
    def channel2(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'channel2' array.array() must have the type code of 'f'"
            self._channel2 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'channel2' field must be a set or sequence and each value of type 'float'"
        self._channel2 = array.array('f', value)

    @property
    def channel3(self):
        """Message field 'channel3'."""
        return self._channel3

    @channel3.setter
    def channel3(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'channel3' array.array() must have the type code of 'f'"
            self._channel3 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'channel3' field must be a set or sequence and each value of type 'float'"
        self._channel3 = array.array('f', value)
