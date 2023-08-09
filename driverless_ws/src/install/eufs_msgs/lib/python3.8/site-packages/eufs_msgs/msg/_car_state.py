# generated from rosidl_generator_py/resource/_idl.py.em
# with input from eufs_msgs:msg/CarState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'linear_acceleration_covariance'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CarState(type):
    """Metaclass of message 'CarState'."""

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
                'eufs_msgs.msg.CarState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__car_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__car_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__car_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__car_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__car_state

            from geometry_msgs.msg import PoseWithCovariance
            if PoseWithCovariance.__class__._TYPE_SUPPORT is None:
                PoseWithCovariance.__class__.__import_type_support__()

            from geometry_msgs.msg import TwistWithCovariance
            if TwistWithCovariance.__class__._TYPE_SUPPORT is None:
                TwistWithCovariance.__class__.__import_type_support__()

            from geometry_msgs.msg import Vector3
            if Vector3.__class__._TYPE_SUPPORT is None:
                Vector3.__class__.__import_type_support__()

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


class CarState(metaclass=Metaclass_CarState):
    """Message class 'CarState'."""

    __slots__ = [
        '_header',
        '_child_frame_id',
        '_pose',
        '_twist',
        '_linear_acceleration',
        '_linear_acceleration_covariance',
        '_slip_angle',
        '_state_of_charge',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'child_frame_id': 'string',
        'pose': 'geometry_msgs/PoseWithCovariance',
        'twist': 'geometry_msgs/TwistWithCovariance',
        'linear_acceleration': 'geometry_msgs/Vector3',
        'linear_acceleration_covariance': 'double[9]',
        'slip_angle': 'double',
        'state_of_charge': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'PoseWithCovariance'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'TwistWithCovariance'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 9),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.child_frame_id = kwargs.get('child_frame_id', str())
        from geometry_msgs.msg import PoseWithCovariance
        self.pose = kwargs.get('pose', PoseWithCovariance())
        from geometry_msgs.msg import TwistWithCovariance
        self.twist = kwargs.get('twist', TwistWithCovariance())
        from geometry_msgs.msg import Vector3
        self.linear_acceleration = kwargs.get('linear_acceleration', Vector3())
        if 'linear_acceleration_covariance' not in kwargs:
            self.linear_acceleration_covariance = numpy.zeros(9, dtype=numpy.float64)
        else:
            self.linear_acceleration_covariance = numpy.array(kwargs.get('linear_acceleration_covariance'), dtype=numpy.float64)
            assert self.linear_acceleration_covariance.shape == (9, )
        self.slip_angle = kwargs.get('slip_angle', float())
        self.state_of_charge = kwargs.get('state_of_charge', float())

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
        if self.child_frame_id != other.child_frame_id:
            return False
        if self.pose != other.pose:
            return False
        if self.twist != other.twist:
            return False
        if self.linear_acceleration != other.linear_acceleration:
            return False
        if all(self.linear_acceleration_covariance != other.linear_acceleration_covariance):
            return False
        if self.slip_angle != other.slip_angle:
            return False
        if self.state_of_charge != other.state_of_charge:
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
    def child_frame_id(self):
        """Message field 'child_frame_id'."""
        return self._child_frame_id

    @child_frame_id.setter
    def child_frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'child_frame_id' field must be of type 'str'"
        self._child_frame_id = value

    @property
    def pose(self):
        """Message field 'pose'."""
        return self._pose

    @pose.setter
    def pose(self, value):
        if __debug__:
            from geometry_msgs.msg import PoseWithCovariance
            assert \
                isinstance(value, PoseWithCovariance), \
                "The 'pose' field must be a sub message of type 'PoseWithCovariance'"
        self._pose = value

    @property
    def twist(self):
        """Message field 'twist'."""
        return self._twist

    @twist.setter
    def twist(self, value):
        if __debug__:
            from geometry_msgs.msg import TwistWithCovariance
            assert \
                isinstance(value, TwistWithCovariance), \
                "The 'twist' field must be a sub message of type 'TwistWithCovariance'"
        self._twist = value

    @property
    def linear_acceleration(self):
        """Message field 'linear_acceleration'."""
        return self._linear_acceleration

    @linear_acceleration.setter
    def linear_acceleration(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'linear_acceleration' field must be a sub message of type 'Vector3'"
        self._linear_acceleration = value

    @property
    def linear_acceleration_covariance(self):
        """Message field 'linear_acceleration_covariance'."""
        return self._linear_acceleration_covariance

    @linear_acceleration_covariance.setter
    def linear_acceleration_covariance(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'linear_acceleration_covariance' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 9, \
                "The 'linear_acceleration_covariance' numpy.ndarray() must have a size of 9"
            self._linear_acceleration_covariance = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 9 and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'linear_acceleration_covariance' field must be a set or sequence with length 9 and each value of type 'float'"
        self._linear_acceleration_covariance = numpy.array(value, dtype=numpy.float64)

    @property
    def slip_angle(self):
        """Message field 'slip_angle'."""
        return self._slip_angle

    @slip_angle.setter
    def slip_angle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'slip_angle' field must be of type 'float'"
        self._slip_angle = value

    @property
    def state_of_charge(self):
        """Message field 'state_of_charge'."""
        return self._state_of_charge

    @state_of_charge.setter
    def state_of_charge(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'state_of_charge' field must be of type 'float'"
        self._state_of_charge = value
