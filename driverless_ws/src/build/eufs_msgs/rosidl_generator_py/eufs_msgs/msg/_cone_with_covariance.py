# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/ConeWithCovariance.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'covariance'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ConeWithCovariance(type):
    """Metaclass of message 'ConeWithCovariance'."""

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
                'eufs_msgs.msg.ConeWithCovariance')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__cone_with_covariance
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__cone_with_covariance
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__cone_with_covariance
            cls._TYPE_SUPPORT = module.type_support_msg__msg__cone_with_covariance
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__cone_with_covariance

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ConeWithCovariance(metaclass=Metaclass_ConeWithCovariance):
    """Message class 'ConeWithCovariance'."""

    __slots__ = [
        '_point',
        '_covariance',
    ]

    _fields_and_field_types = {
        'point': 'geometry_msgs/Point',
        'covariance': 'double[4]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from geometry_msgs.msg import Point
        self.point = kwargs.get('point', Point())
        if 'covariance' not in kwargs:
            self.covariance = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.covariance = numpy.array(kwargs.get('covariance'), dtype=numpy.float64)
            assert self.covariance.shape == (4, )

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
        if self.point != other.point:
            return False
        if all(self.covariance != other.covariance):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def point(self):
        """Message field 'point'."""
        return self._point

    @point.setter
    def point(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'point' field must be a sub message of type 'Point'"
        self._point = value

    @property
    def covariance(self):
        """Message field 'covariance'."""
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'covariance' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'covariance' numpy.ndarray() must have a size of 4"
            self._covariance = value
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
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'covariance' field must be a set or sequence with length 4 and each value of type 'float'"
        self._covariance = numpy.array(value, dtype=numpy.float64)
