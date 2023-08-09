# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/EKFErr.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_EKFErr(type):
    """Metaclass of message 'EKFErr'."""

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
                'eufs_msgs.msg.EKFErr')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__ekf_err
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__ekf_err
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__ekf_err
            cls._TYPE_SUPPORT = module.type_support_msg__msg__ekf_err
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__ekf_err

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


class EKFErr(metaclass=Metaclass_EKFErr):
    """Message class 'EKFErr'."""

    __slots__ = [
        '_header',
        '_gps_x_vel_err',
        '_gps_y_vel_err',
        '_imu_x_acc_err',
        '_imu_y_acc_err',
        '_imu_yaw_err',
        '_ekf_x_vel_var',
        '_ekf_y_vel_var',
        '_ekf_x_acc_var',
        '_ekf_y_acc_var',
        '_ekf_yaw_var',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'gps_x_vel_err': 'double',
        'gps_y_vel_err': 'double',
        'imu_x_acc_err': 'double',
        'imu_y_acc_err': 'double',
        'imu_yaw_err': 'double',
        'ekf_x_vel_var': 'double',
        'ekf_y_vel_var': 'double',
        'ekf_x_acc_var': 'double',
        'ekf_y_acc_var': 'double',
        'ekf_yaw_var': 'double',
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
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.gps_x_vel_err = kwargs.get('gps_x_vel_err', float())
        self.gps_y_vel_err = kwargs.get('gps_y_vel_err', float())
        self.imu_x_acc_err = kwargs.get('imu_x_acc_err', float())
        self.imu_y_acc_err = kwargs.get('imu_y_acc_err', float())
        self.imu_yaw_err = kwargs.get('imu_yaw_err', float())
        self.ekf_x_vel_var = kwargs.get('ekf_x_vel_var', float())
        self.ekf_y_vel_var = kwargs.get('ekf_y_vel_var', float())
        self.ekf_x_acc_var = kwargs.get('ekf_x_acc_var', float())
        self.ekf_y_acc_var = kwargs.get('ekf_y_acc_var', float())
        self.ekf_yaw_var = kwargs.get('ekf_yaw_var', float())

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
        if self.gps_x_vel_err != other.gps_x_vel_err:
            return False
        if self.gps_y_vel_err != other.gps_y_vel_err:
            return False
        if self.imu_x_acc_err != other.imu_x_acc_err:
            return False
        if self.imu_y_acc_err != other.imu_y_acc_err:
            return False
        if self.imu_yaw_err != other.imu_yaw_err:
            return False
        if self.ekf_x_vel_var != other.ekf_x_vel_var:
            return False
        if self.ekf_y_vel_var != other.ekf_y_vel_var:
            return False
        if self.ekf_x_acc_var != other.ekf_x_acc_var:
            return False
        if self.ekf_y_acc_var != other.ekf_y_acc_var:
            return False
        if self.ekf_yaw_var != other.ekf_yaw_var:
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
    def gps_x_vel_err(self):
        """Message field 'gps_x_vel_err'."""
        return self._gps_x_vel_err

    @gps_x_vel_err.setter
    def gps_x_vel_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'gps_x_vel_err' field must be of type 'float'"
        self._gps_x_vel_err = value

    @property
    def gps_y_vel_err(self):
        """Message field 'gps_y_vel_err'."""
        return self._gps_y_vel_err

    @gps_y_vel_err.setter
    def gps_y_vel_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'gps_y_vel_err' field must be of type 'float'"
        self._gps_y_vel_err = value

    @property
    def imu_x_acc_err(self):
        """Message field 'imu_x_acc_err'."""
        return self._imu_x_acc_err

    @imu_x_acc_err.setter
    def imu_x_acc_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'imu_x_acc_err' field must be of type 'float'"
        self._imu_x_acc_err = value

    @property
    def imu_y_acc_err(self):
        """Message field 'imu_y_acc_err'."""
        return self._imu_y_acc_err

    @imu_y_acc_err.setter
    def imu_y_acc_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'imu_y_acc_err' field must be of type 'float'"
        self._imu_y_acc_err = value

    @property
    def imu_yaw_err(self):
        """Message field 'imu_yaw_err'."""
        return self._imu_yaw_err

    @imu_yaw_err.setter
    def imu_yaw_err(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'imu_yaw_err' field must be of type 'float'"
        self._imu_yaw_err = value

    @property
    def ekf_x_vel_var(self):
        """Message field 'ekf_x_vel_var'."""
        return self._ekf_x_vel_var

    @ekf_x_vel_var.setter
    def ekf_x_vel_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ekf_x_vel_var' field must be of type 'float'"
        self._ekf_x_vel_var = value

    @property
    def ekf_y_vel_var(self):
        """Message field 'ekf_y_vel_var'."""
        return self._ekf_y_vel_var

    @ekf_y_vel_var.setter
    def ekf_y_vel_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ekf_y_vel_var' field must be of type 'float'"
        self._ekf_y_vel_var = value

    @property
    def ekf_x_acc_var(self):
        """Message field 'ekf_x_acc_var'."""
        return self._ekf_x_acc_var

    @ekf_x_acc_var.setter
    def ekf_x_acc_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ekf_x_acc_var' field must be of type 'float'"
        self._ekf_x_acc_var = value

    @property
    def ekf_y_acc_var(self):
        """Message field 'ekf_y_acc_var'."""
        return self._ekf_y_acc_var

    @ekf_y_acc_var.setter
    def ekf_y_acc_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ekf_y_acc_var' field must be of type 'float'"
        self._ekf_y_acc_var = value

    @property
    def ekf_yaw_var(self):
        """Message field 'ekf_yaw_var'."""
        return self._ekf_yaw_var

    @ekf_yaw_var.setter
    def ekf_yaw_var(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ekf_yaw_var' field must be of type 'float'"
        self._ekf_yaw_var = value
