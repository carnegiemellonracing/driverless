# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/CanState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CanState(type):
    """Metaclass of message 'CanState'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'AS_OFF': 0,
        'AS_READY': 1,
        'AS_DRIVING': 2,
        'AS_EMERGENCY_BRAKE': 3,
        'AS_FINISHED': 4,
        'AMI_NOT_SELECTED': 10,
        'AMI_ACCELERATION': 11,
        'AMI_SKIDPAD': 12,
        'AMI_AUTOCROSS': 13,
        'AMI_TRACK_DRIVE': 14,
        'AMI_AUTONOMOUS_DEMO': 15,
        'AMI_ADS_INSPECTION': 16,
        'AMI_ADS_EBS': 17,
        'AMI_DDT_INSPECTION_A': 18,
        'AMI_DDT_INSPECTION_B': 19,
        'AMI_JOYSTICK': 20,
        'AMI_MANUAL': 21,
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
                'eufs_msgs.msg.CanState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__can_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__can_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__can_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__can_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__can_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'AS_OFF': cls.__constants['AS_OFF'],
            'AS_READY': cls.__constants['AS_READY'],
            'AS_DRIVING': cls.__constants['AS_DRIVING'],
            'AS_EMERGENCY_BRAKE': cls.__constants['AS_EMERGENCY_BRAKE'],
            'AS_FINISHED': cls.__constants['AS_FINISHED'],
            'AMI_NOT_SELECTED': cls.__constants['AMI_NOT_SELECTED'],
            'AMI_ACCELERATION': cls.__constants['AMI_ACCELERATION'],
            'AMI_SKIDPAD': cls.__constants['AMI_SKIDPAD'],
            'AMI_AUTOCROSS': cls.__constants['AMI_AUTOCROSS'],
            'AMI_TRACK_DRIVE': cls.__constants['AMI_TRACK_DRIVE'],
            'AMI_AUTONOMOUS_DEMO': cls.__constants['AMI_AUTONOMOUS_DEMO'],
            'AMI_ADS_INSPECTION': cls.__constants['AMI_ADS_INSPECTION'],
            'AMI_ADS_EBS': cls.__constants['AMI_ADS_EBS'],
            'AMI_DDT_INSPECTION_A': cls.__constants['AMI_DDT_INSPECTION_A'],
            'AMI_DDT_INSPECTION_B': cls.__constants['AMI_DDT_INSPECTION_B'],
            'AMI_JOYSTICK': cls.__constants['AMI_JOYSTICK'],
            'AMI_MANUAL': cls.__constants['AMI_MANUAL'],
        }

    @property
    def AS_OFF(self):
        """Message constant 'AS_OFF'."""
        return Metaclass_CanState.__constants['AS_OFF']

    @property
    def AS_READY(self):
        """Message constant 'AS_READY'."""
        return Metaclass_CanState.__constants['AS_READY']

    @property
    def AS_DRIVING(self):
        """Message constant 'AS_DRIVING'."""
        return Metaclass_CanState.__constants['AS_DRIVING']

    @property
    def AS_EMERGENCY_BRAKE(self):
        """Message constant 'AS_EMERGENCY_BRAKE'."""
        return Metaclass_CanState.__constants['AS_EMERGENCY_BRAKE']

    @property
    def AS_FINISHED(self):
        """Message constant 'AS_FINISHED'."""
        return Metaclass_CanState.__constants['AS_FINISHED']

    @property
    def AMI_NOT_SELECTED(self):
        """Message constant 'AMI_NOT_SELECTED'."""
        return Metaclass_CanState.__constants['AMI_NOT_SELECTED']

    @property
    def AMI_ACCELERATION(self):
        """Message constant 'AMI_ACCELERATION'."""
        return Metaclass_CanState.__constants['AMI_ACCELERATION']

    @property
    def AMI_SKIDPAD(self):
        """Message constant 'AMI_SKIDPAD'."""
        return Metaclass_CanState.__constants['AMI_SKIDPAD']

    @property
    def AMI_AUTOCROSS(self):
        """Message constant 'AMI_AUTOCROSS'."""
        return Metaclass_CanState.__constants['AMI_AUTOCROSS']

    @property
    def AMI_TRACK_DRIVE(self):
        """Message constant 'AMI_TRACK_DRIVE'."""
        return Metaclass_CanState.__constants['AMI_TRACK_DRIVE']

    @property
    def AMI_AUTONOMOUS_DEMO(self):
        """Message constant 'AMI_AUTONOMOUS_DEMO'."""
        return Metaclass_CanState.__constants['AMI_AUTONOMOUS_DEMO']

    @property
    def AMI_ADS_INSPECTION(self):
        """Message constant 'AMI_ADS_INSPECTION'."""
        return Metaclass_CanState.__constants['AMI_ADS_INSPECTION']

    @property
    def AMI_ADS_EBS(self):
        """Message constant 'AMI_ADS_EBS'."""
        return Metaclass_CanState.__constants['AMI_ADS_EBS']

    @property
    def AMI_DDT_INSPECTION_A(self):
        """Message constant 'AMI_DDT_INSPECTION_A'."""
        return Metaclass_CanState.__constants['AMI_DDT_INSPECTION_A']

    @property
    def AMI_DDT_INSPECTION_B(self):
        """Message constant 'AMI_DDT_INSPECTION_B'."""
        return Metaclass_CanState.__constants['AMI_DDT_INSPECTION_B']

    @property
    def AMI_JOYSTICK(self):
        """Message constant 'AMI_JOYSTICK'."""
        return Metaclass_CanState.__constants['AMI_JOYSTICK']

    @property
    def AMI_MANUAL(self):
        """Message constant 'AMI_MANUAL'."""
        return Metaclass_CanState.__constants['AMI_MANUAL']


class CanState(metaclass=Metaclass_CanState):
    """
    Message class 'CanState'.

    Constants:
      AS_OFF
      AS_READY
      AS_DRIVING
      AS_EMERGENCY_BRAKE
      AS_FINISHED
      AMI_NOT_SELECTED
      AMI_ACCELERATION
      AMI_SKIDPAD
      AMI_AUTOCROSS
      AMI_TRACK_DRIVE
      AMI_AUTONOMOUS_DEMO
      AMI_ADS_INSPECTION
      AMI_ADS_EBS
      AMI_DDT_INSPECTION_A
      AMI_DDT_INSPECTION_B
      AMI_JOYSTICK
      AMI_MANUAL
    """

    __slots__ = [
        '_as_state',
        '_ami_state',
    ]

    _fields_and_field_types = {
        'as_state': 'uint16',
        'ami_state': 'uint16',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.as_state = kwargs.get('as_state', int())
        self.ami_state = kwargs.get('ami_state', int())

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
        if self.as_state != other.as_state:
            return False
        if self.ami_state != other.ami_state:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def as_state(self):
        """Message field 'as_state'."""
        return self._as_state

    @as_state.setter
    def as_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'as_state' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'as_state' field must be an unsigned integer in [0, 65535]"
        self._as_state = value

    @property
    def ami_state(self):
        """Message field 'ami_state'."""
        return self._ami_state

    @ami_state.setter
    def ami_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ami_state' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'ami_state' field must be an unsigned integer in [0, 65535]"
        self._ami_state = value
