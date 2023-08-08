# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/TopicStatus.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_TopicStatus(type):
    """Metaclass of message 'TopicStatus'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'OFF': 0,
        'PUBLISHING': 1,
        'TIMEOUT_EXCEEDED': 2,
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
                'eufs_msgs.msg.TopicStatus')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__topic_status
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__topic_status
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__topic_status
            cls._TYPE_SUPPORT = module.type_support_msg__msg__topic_status
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__topic_status

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'OFF': cls.__constants['OFF'],
            'PUBLISHING': cls.__constants['PUBLISHING'],
            'TIMEOUT_EXCEEDED': cls.__constants['TIMEOUT_EXCEEDED'],
        }

    @property
    def OFF(self):
        """Message constant 'OFF'."""
        return Metaclass_TopicStatus.__constants['OFF']

    @property
    def PUBLISHING(self):
        """Message constant 'PUBLISHING'."""
        return Metaclass_TopicStatus.__constants['PUBLISHING']

    @property
    def TIMEOUT_EXCEEDED(self):
        """Message constant 'TIMEOUT_EXCEEDED'."""
        return Metaclass_TopicStatus.__constants['TIMEOUT_EXCEEDED']


class TopicStatus(metaclass=Metaclass_TopicStatus):
    """
    Message class 'TopicStatus'.

    Constants:
      OFF
      PUBLISHING
      TIMEOUT_EXCEEDED
    """

    __slots__ = [
        '_topic',
        '_description',
        '_group',
        '_trigger_ebs',
        '_log_level',
        '_status',
    ]

    _fields_and_field_types = {
        'topic': 'string',
        'description': 'string',
        'group': 'string',
        'trigger_ebs': 'boolean',
        'log_level': 'string',
        'status': 'uint16',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.topic = kwargs.get('topic', str())
        self.description = kwargs.get('description', str())
        self.group = kwargs.get('group', str())
        self.trigger_ebs = kwargs.get('trigger_ebs', bool())
        self.log_level = kwargs.get('log_level', str())
        self.status = kwargs.get('status', int())

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
        if self.topic != other.topic:
            return False
        if self.description != other.description:
            return False
        if self.group != other.group:
            return False
        if self.trigger_ebs != other.trigger_ebs:
            return False
        if self.log_level != other.log_level:
            return False
        if self.status != other.status:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def topic(self):
        """Message field 'topic'."""
        return self._topic

    @topic.setter
    def topic(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'topic' field must be of type 'str'"
        self._topic = value

    @property
    def description(self):
        """Message field 'description'."""
        return self._description

    @description.setter
    def description(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'description' field must be of type 'str'"
        self._description = value

    @property
    def group(self):
        """Message field 'group'."""
        return self._group

    @group.setter
    def group(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'group' field must be of type 'str'"
        self._group = value

    @property
    def trigger_ebs(self):
        """Message field 'trigger_ebs'."""
        return self._trigger_ebs

    @trigger_ebs.setter
    def trigger_ebs(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'trigger_ebs' field must be of type 'bool'"
        self._trigger_ebs = value

    @property
    def log_level(self):
        """Message field 'log_level'."""
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'log_level' field must be of type 'str'"
        self._log_level = value

    @property
    def status(self):
        """Message field 'status'."""
        return self._status

    @status.setter
    def status(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'status' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'status' field must be an unsigned integer in [0, 65535]"
        self._status = value
