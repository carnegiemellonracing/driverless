# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/NodeState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_NodeState(type):
    """Metaclass of message 'NodeState'."""

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
                'eufs_msgs.msg.NodeState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__node_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__node_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__node_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__node_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__node_state

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class NodeState(metaclass=Metaclass_NodeState):
    """Message class 'NodeState'."""

    __slots__ = [
        '_id',
        '_name',
        '_exp_heartbeat',
        '_last_heartbeat',
        '_severity',
        '_online',
    ]

    _fields_and_field_types = {
        'id': 'uint16',
        'name': 'string',
        'exp_heartbeat': 'uint8',
        'last_heartbeat': 'builtin_interfaces/Time',
        'severity': 'uint8',
        'online': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.id = kwargs.get('id', int())
        self.name = kwargs.get('name', str())
        self.exp_heartbeat = kwargs.get('exp_heartbeat', int())
        from builtin_interfaces.msg import Time
        self.last_heartbeat = kwargs.get('last_heartbeat', Time())
        self.severity = kwargs.get('severity', int())
        self.online = kwargs.get('online', bool())

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
        if self.id != other.id:
            return False
        if self.name != other.name:
            return False
        if self.exp_heartbeat != other.exp_heartbeat:
            return False
        if self.last_heartbeat != other.last_heartbeat:
            return False
        if self.severity != other.severity:
            return False
        if self.online != other.online:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property  # noqa: A003
    def id(self):  # noqa: A003
        """Message field 'id'."""
        return self._id

    @id.setter  # noqa: A003
    def id(self, value):  # noqa: A003
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'id' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'id' field must be an unsigned integer in [0, 65535]"
        self._id = value

    @property
    def name(self):
        """Message field 'name'."""
        return self._name

    @name.setter
    def name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'name' field must be of type 'str'"
        self._name = value

    @property
    def exp_heartbeat(self):
        """Message field 'exp_heartbeat'."""
        return self._exp_heartbeat

    @exp_heartbeat.setter
    def exp_heartbeat(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'exp_heartbeat' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'exp_heartbeat' field must be an unsigned integer in [0, 255]"
        self._exp_heartbeat = value

    @property
    def last_heartbeat(self):
        """Message field 'last_heartbeat'."""
        return self._last_heartbeat

    @last_heartbeat.setter
    def last_heartbeat(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'last_heartbeat' field must be a sub message of type 'Time'"
        self._last_heartbeat = value

    @property
    def severity(self):
        """Message field 'severity'."""
        return self._severity

    @severity.setter
    def severity(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'severity' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'severity' field must be an unsigned integer in [0, 255]"
        self._severity = value

    @property
    def online(self):
        """Message field 'online'."""
        return self._online

    @online.setter
    def online(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'online' field must be of type 'bool'"
        self._online = value
