# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/SLAMErr.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SLAMErr(type):
    """Metaclass of message 'SLAMErr'."""

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
                'eufs_msgs.msg.SLAMErr')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__slam_err
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__slam_err
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__slam_err
            cls._TYPE_SUPPORT = module.type_support_msg__msg__slam_err
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__slam_err

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


class SLAMErr(metaclass=Metaclass_SLAMErr):
    """Message class 'SLAMErr'."""

    __slots__ = [
        '_header',
        '_x_err',
        '_y_err',
        '_z_err',
        '_x_orient_err',
        '_y_orient_err',
        '_z_orient_err',
        '_w_orient_err',
        '_map_similarity',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'x_err': 'double',
        'y_err': 'double',
        'z_err': 'double',
        'x_orient_err': 'double',
        'y_orient_err': 'double',
        'z_orient_err': 'double',
        'w_orient_err': 'double',
        'map_similarity': 'double',
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
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.x_err = kwargs.get('x_err', float())
        self.y_err = kwargs.get('y_err', float())
        self.z_err = kwargs.get('z_err', float())
        self.x_orient_err = kwargs.get('x_orient_err', float())
        self.y_orient_err = kwargs.get('y_orient_err', float())
        self.z_orient_err = kwargs.get('z_orient_err', float())
        self.w_orient_err = kwargs.get('w_orient_err', float())
        self.map_similarity = kwargs.get('map_similarity', float())

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
        if self.x_err != other.x_err:
            return False
        if self.y_err != other.y_err:
            return False
        if self.z_err != other.z_err:
            return False
        if self.x_orient_err != other.x_orient_err:
            return False
        if self.y_orient_err != other.y_orient_err:
            return False
        if self.z_orient_err != other.z_orient_err:
            return False
        if self.w_orient_err != other.w_orient_err:
            return False
        if self.map_similarity != other.map_similarity:
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
    def x_err(self):
        """Message field 'x_err'."""
        return self._x_err

    @x_err.setter
    def x_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x_err' field must be of type 'float'"
        self._x_err = value

    @property
    def y_err(self):
        """Message field 'y_err'."""
        return self._y_err

    @y_err.setter
    def y_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y_err' field must be of type 'float'"
        self._y_err = value

    @property
    def z_err(self):
        """Message field 'z_err'."""
        return self._z_err

    @z_err.setter
    def z_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z_err' field must be of type 'float'"
        self._z_err = value

    @property
    def x_orient_err(self):
        """Message field 'x_orient_err'."""
        return self._x_orient_err

    @x_orient_err.setter
    def x_orient_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x_orient_err' field must be of type 'float'"
        self._x_orient_err = value

    @property
    def y_orient_err(self):
        """Message field 'y_orient_err'."""
        return self._y_orient_err

    @y_orient_err.setter
    def y_orient_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y_orient_err' field must be of type 'float'"
        self._y_orient_err = value

    @property
    def z_orient_err(self):
        """Message field 'z_orient_err'."""
        return self._z_orient_err

    @z_orient_err.setter
    def z_orient_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z_orient_err' field must be of type 'float'"
        self._z_orient_err = value

    @property
    def w_orient_err(self):
        """Message field 'w_orient_err'."""
        return self._w_orient_err

    @w_orient_err.setter
    def w_orient_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'w_orient_err' field must be of type 'float'"
        self._w_orient_err = value

    @property
    def map_similarity(self):
        """Message field 'map_similarity'."""
        return self._map_similarity

    @map_similarity.setter
    def map_similarity(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'map_similarity' field must be of type 'float'"
        self._map_similarity = value
