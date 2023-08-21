# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/IntegrationErr.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_IntegrationErr(type):
    """Metaclass of message 'IntegrationErr'."""

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
                'eufs_msgs.msg.IntegrationErr')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__integration_err
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__integration_err
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__integration_err
            cls._TYPE_SUPPORT = module.type_support_msg__msg__integration_err
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__integration_err

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


class IntegrationErr(metaclass=Metaclass_IntegrationErr):
    """Message class 'IntegrationErr'."""

    __slots__ = [
        '_header',
        '_position_err',
        '_orientation_err',
        '_linear_vel_err',
        '_angular_vel_err',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'position_err': 'double',
        'orientation_err': 'double',
        'linear_vel_err': 'double',
        'angular_vel_err': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
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
        self.position_err = kwargs.get('position_err', float())
        self.orientation_err = kwargs.get('orientation_err', float())
        self.linear_vel_err = kwargs.get('linear_vel_err', float())
        self.angular_vel_err = kwargs.get('angular_vel_err', float())

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
        if self.position_err != other.position_err:
            return False
        if self.orientation_err != other.orientation_err:
            return False
        if self.linear_vel_err != other.linear_vel_err:
            return False
        if self.angular_vel_err != other.angular_vel_err:
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
    def position_err(self):
        """Message field 'position_err'."""
        return self._position_err

    @position_err.setter
    def position_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'position_err' field must be of type 'float'"
        self._position_err = value

    @property
    def orientation_err(self):
        """Message field 'orientation_err'."""
        return self._orientation_err

    @orientation_err.setter
    def orientation_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'orientation_err' field must be of type 'float'"
        self._orientation_err = value

    @property
    def linear_vel_err(self):
        """Message field 'linear_vel_err'."""
        return self._linear_vel_err

    @linear_vel_err.setter
    def linear_vel_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'linear_vel_err' field must be of type 'float'"
        self._linear_vel_err = value

    @property
    def angular_vel_err(self):
        """Message field 'angular_vel_err'."""
        return self._angular_vel_err

    @angular_vel_err.setter
    def angular_vel_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'angular_vel_err' field must be of type 'float'"
        self._angular_vel_err = value
