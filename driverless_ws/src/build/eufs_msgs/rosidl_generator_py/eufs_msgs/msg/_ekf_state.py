# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/EKFState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_EKFState(type):
    """Metaclass of message 'EKFState'."""

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
                'eufs_msgs.msg.EKFState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__ekf_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__ekf_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__ekf_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__ekf_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__ekf_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class EKFState(metaclass=Metaclass_EKFState):
    """Message class 'EKFState'."""

    __slots__ = [
        '_gps_received',
        '_imu_received',
        '_wheel_odom_received',
        '_ekf_odom_received',
        '_ekf_accel_received',
        '_currently_over_covariance_limit',
        '_consecutive_turns_over_covariance_limit',
        '_recommends_failure',
    ]

    _fields_and_field_types = {
        'gps_received': 'boolean',
        'imu_received': 'boolean',
        'wheel_odom_received': 'boolean',
        'ekf_odom_received': 'boolean',
        'ekf_accel_received': 'boolean',
        'currently_over_covariance_limit': 'boolean',
        'consecutive_turns_over_covariance_limit': 'boolean',
        'recommends_failure': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.gps_received = kwargs.get('gps_received', bool())
        self.imu_received = kwargs.get('imu_received', bool())
        self.wheel_odom_received = kwargs.get('wheel_odom_received', bool())
        self.ekf_odom_received = kwargs.get('ekf_odom_received', bool())
        self.ekf_accel_received = kwargs.get('ekf_accel_received', bool())
        self.currently_over_covariance_limit = kwargs.get('currently_over_covariance_limit', bool())
        self.consecutive_turns_over_covariance_limit = kwargs.get('consecutive_turns_over_covariance_limit', bool())
        self.recommends_failure = kwargs.get('recommends_failure', bool())

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
        if self.gps_received != other.gps_received:
            return False
        if self.imu_received != other.imu_received:
            return False
        if self.wheel_odom_received != other.wheel_odom_received:
            return False
        if self.ekf_odom_received != other.ekf_odom_received:
            return False
        if self.ekf_accel_received != other.ekf_accel_received:
            return False
        if self.currently_over_covariance_limit != other.currently_over_covariance_limit:
            return False
        if self.consecutive_turns_over_covariance_limit != other.consecutive_turns_over_covariance_limit:
            return False
        if self.recommends_failure != other.recommends_failure:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def gps_received(self):
        """Message field 'gps_received'."""
        return self._gps_received

    @gps_received.setter
    def gps_received(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'gps_received' field must be of type 'bool'"
        self._gps_received = value

    @property
    def imu_received(self):
        """Message field 'imu_received'."""
        return self._imu_received

    @imu_received.setter
    def imu_received(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'imu_received' field must be of type 'bool'"
        self._imu_received = value

    @property
    def wheel_odom_received(self):
        """Message field 'wheel_odom_received'."""
        return self._wheel_odom_received

    @wheel_odom_received.setter
    def wheel_odom_received(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'wheel_odom_received' field must be of type 'bool'"
        self._wheel_odom_received = value

    @property
    def ekf_odom_received(self):
        """Message field 'ekf_odom_received'."""
        return self._ekf_odom_received

    @ekf_odom_received.setter
    def ekf_odom_received(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'ekf_odom_received' field must be of type 'bool'"
        self._ekf_odom_received = value

    @property
    def ekf_accel_received(self):
        """Message field 'ekf_accel_received'."""
        return self._ekf_accel_received

    @ekf_accel_received.setter
    def ekf_accel_received(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'ekf_accel_received' field must be of type 'bool'"
        self._ekf_accel_received = value

    @property
    def currently_over_covariance_limit(self):
        """Message field 'currently_over_covariance_limit'."""
        return self._currently_over_covariance_limit

    @currently_over_covariance_limit.setter
    def currently_over_covariance_limit(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'currently_over_covariance_limit' field must be of type 'bool'"
        self._currently_over_covariance_limit = value

    @property
    def consecutive_turns_over_covariance_limit(self):
        """Message field 'consecutive_turns_over_covariance_limit'."""
        return self._consecutive_turns_over_covariance_limit

    @consecutive_turns_over_covariance_limit.setter
    def consecutive_turns_over_covariance_limit(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'consecutive_turns_over_covariance_limit' field must be of type 'bool'"
        self._consecutive_turns_over_covariance_limit = value

    @property
    def recommends_failure(self):
        """Message field 'recommends_failure'."""
        return self._recommends_failure

    @recommends_failure.setter
    def recommends_failure(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'recommends_failure' field must be of type 'bool'"
        self._recommends_failure = value
