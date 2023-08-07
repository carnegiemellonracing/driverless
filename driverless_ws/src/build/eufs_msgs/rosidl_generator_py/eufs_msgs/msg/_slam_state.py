# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/SLAMState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SLAMState(type):
    """Metaclass of message 'SLAMState'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'NO_INPUTS': 0,
        'MAPPING': 1,
        'LOCALISING': 2,
        'TIMEOUT': 3,
        'RECOMMENDS_FAILURE': 4,
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
                'eufs_msgs.msg.SLAMState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__slam_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__slam_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__slam_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__slam_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__slam_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'NO_INPUTS': cls.__constants['NO_INPUTS'],
            'MAPPING': cls.__constants['MAPPING'],
            'LOCALISING': cls.__constants['LOCALISING'],
            'TIMEOUT': cls.__constants['TIMEOUT'],
            'RECOMMENDS_FAILURE': cls.__constants['RECOMMENDS_FAILURE'],
        }

    @property
    def NO_INPUTS(self):
        """Message constant 'NO_INPUTS'."""
        return Metaclass_SLAMState.__constants['NO_INPUTS']

    @property
    def MAPPING(self):
        """Message constant 'MAPPING'."""
        return Metaclass_SLAMState.__constants['MAPPING']

    @property
    def LOCALISING(self):
        """Message constant 'LOCALISING'."""
        return Metaclass_SLAMState.__constants['LOCALISING']

    @property
    def TIMEOUT(self):
        """Message constant 'TIMEOUT'."""
        return Metaclass_SLAMState.__constants['TIMEOUT']

    @property
    def RECOMMENDS_FAILURE(self):
        """Message constant 'RECOMMENDS_FAILURE'."""
        return Metaclass_SLAMState.__constants['RECOMMENDS_FAILURE']


class SLAMState(metaclass=Metaclass_SLAMState):
    """
    Message class 'SLAMState'.

    Constants:
      NO_INPUTS
      MAPPING
      LOCALISING
      TIMEOUT
      RECOMMENDS_FAILURE
    """

    __slots__ = [
        '_loop_closed',
        '_laps',
        '_status',
        '_state',
    ]

    _fields_and_field_types = {
        'loop_closed': 'boolean',
        'laps': 'int16',
        'status': 'string',
        'state': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.loop_closed = kwargs.get('loop_closed', bool())
        self.laps = kwargs.get('laps', int())
        self.status = kwargs.get('status', str())
        self.state = kwargs.get('state', int())

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
        if self.loop_closed != other.loop_closed:
            return False
        if self.laps != other.laps:
            return False
        if self.status != other.status:
            return False
        if self.state != other.state:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def loop_closed(self):
        """Message field 'loop_closed'."""
        return self._loop_closed

    @loop_closed.setter
    def loop_closed(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'loop_closed' field must be of type 'bool'"
        self._loop_closed = value

    @property
    def laps(self):
        """Message field 'laps'."""
        return self._laps

    @laps.setter
    def laps(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'laps' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'laps' field must be an integer in [-32768, 32767]"
        self._laps = value

    @property
    def status(self):
        """Message field 'status'."""
        return self._status

    @status.setter
    def status(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'status' field must be of type 'str'"
        self._status = value

    @property
    def state(self):
        """Message field 'state'."""
        return self._state

    @state.setter
    def state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'state' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'state' field must be an integer in [-128, 127]"
        self._state = value
