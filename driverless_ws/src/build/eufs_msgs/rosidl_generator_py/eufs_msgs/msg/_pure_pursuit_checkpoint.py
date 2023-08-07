# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PurePursuitCheckpoint(type):
    """Metaclass of message 'PurePursuitCheckpoint'."""

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
                'eufs_msgs.msg.PurePursuitCheckpoint')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__pure_pursuit_checkpoint
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__pure_pursuit_checkpoint
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__pure_pursuit_checkpoint
            cls._TYPE_SUPPORT = module.type_support_msg__msg__pure_pursuit_checkpoint
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__pure_pursuit_checkpoint

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


class PurePursuitCheckpoint(metaclass=Metaclass_PurePursuitCheckpoint):
    """Message class 'PurePursuitCheckpoint'."""

    __slots__ = [
        '_position',
        '_max_speed',
        '_max_lateral_acceleration',
    ]

    _fields_and_field_types = {
        'position': 'geometry_msgs/Point',
        'max_speed': 'double',
        'max_lateral_acceleration': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from geometry_msgs.msg import Point
        self.position = kwargs.get('position', Point())
        self.max_speed = kwargs.get('max_speed', float())
        self.max_lateral_acceleration = kwargs.get('max_lateral_acceleration', float())

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
        if self.position != other.position:
            return False
        if self.max_speed != other.max_speed:
            return False
        if self.max_lateral_acceleration != other.max_lateral_acceleration:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'position' field must be a sub message of type 'Point'"
        self._position = value

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
    def max_lateral_acceleration(self):
        """Message field 'max_lateral_acceleration'."""
        return self._max_lateral_acceleration

    @max_lateral_acceleration.setter
    def max_lateral_acceleration(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'max_lateral_acceleration' field must be of type 'float'"
        self._max_lateral_acceleration = value
