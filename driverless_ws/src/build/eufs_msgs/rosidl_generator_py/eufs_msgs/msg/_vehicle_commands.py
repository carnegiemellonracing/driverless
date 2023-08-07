# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/VehicleCommands.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_VehicleCommands(type):
    """Metaclass of message 'VehicleCommands'."""

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
                'eufs_msgs.msg.VehicleCommands')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__vehicle_commands
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__vehicle_commands
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__vehicle_commands
            cls._TYPE_SUPPORT = module.type_support_msg__msg__vehicle_commands
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__vehicle_commands

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class VehicleCommands(metaclass=Metaclass_VehicleCommands):
    """Message class 'VehicleCommands'."""

    __slots__ = [
        '_handshake',
        '_ebs',
        '_direction',
        '_mission_status',
        '_braking',
        '_torque',
        '_steering',
        '_rpm',
    ]

    _fields_and_field_types = {
        'handshake': 'int8',
        'ebs': 'int8',
        'direction': 'int8',
        'mission_status': 'int8',
        'braking': 'double',
        'torque': 'double',
        'steering': 'double',
        'rpm': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.handshake = kwargs.get('handshake', int())
        self.ebs = kwargs.get('ebs', int())
        self.direction = kwargs.get('direction', int())
        self.mission_status = kwargs.get('mission_status', int())
        self.braking = kwargs.get('braking', float())
        self.torque = kwargs.get('torque', float())
        self.steering = kwargs.get('steering', float())
        self.rpm = kwargs.get('rpm', float())

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
        if self.handshake != other.handshake:
            return False
        if self.ebs != other.ebs:
            return False
        if self.direction != other.direction:
            return False
        if self.mission_status != other.mission_status:
            return False
        if self.braking != other.braking:
            return False
        if self.torque != other.torque:
            return False
        if self.steering != other.steering:
            return False
        if self.rpm != other.rpm:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def handshake(self):
        """Message field 'handshake'."""
        return self._handshake

    @handshake.setter
    def handshake(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'handshake' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'handshake' field must be an integer in [-128, 127]"
        self._handshake = value

    @property
    def ebs(self):
        """Message field 'ebs'."""
        return self._ebs

    @ebs.setter
    def ebs(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ebs' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'ebs' field must be an integer in [-128, 127]"
        self._ebs = value

    @property
    def direction(self):
        """Message field 'direction'."""
        return self._direction

    @direction.setter
    def direction(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'direction' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'direction' field must be an integer in [-128, 127]"
        self._direction = value

    @property
    def mission_status(self):
        """Message field 'mission_status'."""
        return self._mission_status

    @mission_status.setter
    def mission_status(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mission_status' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'mission_status' field must be an integer in [-128, 127]"
        self._mission_status = value

    @property
    def braking(self):
        """Message field 'braking'."""
        return self._braking

    @braking.setter
    def braking(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'braking' field must be of type 'float'"
        self._braking = value

    @property
    def torque(self):
        """Message field 'torque'."""
        return self._torque

    @torque.setter
    def torque(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'torque' field must be of type 'float'"
        self._torque = value

    @property
    def steering(self):
        """Message field 'steering'."""
        return self._steering

    @steering.setter
    def steering(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'steering' field must be of type 'float'"
        self._steering = value

    @property
    def rpm(self):
        """Message field 'rpm'."""
        return self._rpm

    @rpm.setter
    def rpm(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rpm' field must be of type 'float'"
        self._rpm = value
