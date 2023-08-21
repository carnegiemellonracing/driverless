# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ConeArrayWithCovariance(type):
    """Metaclass of message 'ConeArrayWithCovariance'."""

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
                'eufs_msgs.msg.ConeArrayWithCovariance')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__cone_array_with_covariance
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__cone_array_with_covariance
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__cone_array_with_covariance
            cls._TYPE_SUPPORT = module.type_support_msg__msg__cone_array_with_covariance
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__cone_array_with_covariance

            from eufs_msgs.msg import ConeWithCovariance
            if ConeWithCovariance.__class__._TYPE_SUPPORT is None:
                ConeWithCovariance.__class__.__import_type_support__()

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


class ConeArrayWithCovariance(metaclass=Metaclass_ConeArrayWithCovariance):
    """Message class 'ConeArrayWithCovariance'."""

    __slots__ = [
        '_header',
        '_blue_cones',
        '_yellow_cones',
        '_orange_cones',
        '_big_orange_cones',
        '_unknown_color_cones',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'blue_cones': 'sequence<eufs_msgs/ConeWithCovariance>',
        'yellow_cones': 'sequence<eufs_msgs/ConeWithCovariance>',
        'orange_cones': 'sequence<eufs_msgs/ConeWithCovariance>',
        'big_orange_cones': 'sequence<eufs_msgs/ConeWithCovariance>',
        'unknown_color_cones': 'sequence<eufs_msgs/ConeWithCovariance>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['eufs_msgs', 'msg'], 'ConeWithCovariance')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['eufs_msgs', 'msg'], 'ConeWithCovariance')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['eufs_msgs', 'msg'], 'ConeWithCovariance')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['eufs_msgs', 'msg'], 'ConeWithCovariance')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['eufs_msgs', 'msg'], 'ConeWithCovariance')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.blue_cones = kwargs.get('blue_cones', [])
        self.yellow_cones = kwargs.get('yellow_cones', [])
        self.orange_cones = kwargs.get('orange_cones', [])
        self.big_orange_cones = kwargs.get('big_orange_cones', [])
        self.unknown_color_cones = kwargs.get('unknown_color_cones', [])

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
        if self.blue_cones != other.blue_cones:
            return False
        if self.yellow_cones != other.yellow_cones:
            return False
        if self.orange_cones != other.orange_cones:
            return False
        if self.big_orange_cones != other.big_orange_cones:
            return False
        if self.unknown_color_cones != other.unknown_color_cones:
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
    def blue_cones(self):
        """Message field 'blue_cones'."""
        return self._blue_cones

    @blue_cones.setter
    def blue_cones(self, value):
        if __debug__:
            from eufs_msgs.msg import ConeWithCovariance
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
                 all(isinstance(v, ConeWithCovariance) for v in value) and
                 True), \
                "The 'blue_cones' field must be a set or sequence and each value of type 'ConeWithCovariance'"
        self._blue_cones = value

    @property
    def yellow_cones(self):
        """Message field 'yellow_cones'."""
        return self._yellow_cones

    @yellow_cones.setter
    def yellow_cones(self, value):
        if __debug__:
            from eufs_msgs.msg import ConeWithCovariance
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
                 all(isinstance(v, ConeWithCovariance) for v in value) and
                 True), \
                "The 'yellow_cones' field must be a set or sequence and each value of type 'ConeWithCovariance'"
        self._yellow_cones = value

    @property
    def orange_cones(self):
        """Message field 'orange_cones'."""
        return self._orange_cones

    @orange_cones.setter
    def orange_cones(self, value):
        if __debug__:
            from eufs_msgs.msg import ConeWithCovariance
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
                 all(isinstance(v, ConeWithCovariance) for v in value) and
                 True), \
                "The 'orange_cones' field must be a set or sequence and each value of type 'ConeWithCovariance'"
        self._orange_cones = value

    @property
    def big_orange_cones(self):
        """Message field 'big_orange_cones'."""
        return self._big_orange_cones

    @big_orange_cones.setter
    def big_orange_cones(self, value):
        if __debug__:
            from eufs_msgs.msg import ConeWithCovariance
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
                 all(isinstance(v, ConeWithCovariance) for v in value) and
                 True), \
                "The 'big_orange_cones' field must be a set or sequence and each value of type 'ConeWithCovariance'"
        self._big_orange_cones = value

    @property
    def unknown_color_cones(self):
        """Message field 'unknown_color_cones'."""
        return self._unknown_color_cones

    @unknown_color_cones.setter
    def unknown_color_cones(self, value):
        if __debug__:
            from eufs_msgs.msg import ConeWithCovariance
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
                 all(isinstance(v, ConeWithCovariance) for v in value) and
                 True), \
                "The 'unknown_color_cones' field must be a set or sequence and each value of type 'ConeWithCovariance'"
        self._unknown_color_cones = value
