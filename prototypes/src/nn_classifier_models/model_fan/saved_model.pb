��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
*
Erf
x"T
y"T"
Ttype:
2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.12v2.6.0-101-g3aa40c3ce9d8��
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:*
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
 Adam/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/layer_normalization/gamma/m
�
4Adam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/m*
_output_shapes
:*
dtype0
�
Adam/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/layer_normalization/beta/m
�
3Adam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/m*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
 Adam/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/layer_normalization/gamma/v
�
4Adam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/v*
_output_shapes
:*
dtype0
�
Adam/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/layer_normalization/beta/v
�
3Adam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
q
axis
	gamma
beta
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
�
iter

beta_1

beta_2
	decay
 learning_ratem:m;m<m=v>v?v@vA
 

0
1
2
3

0
1
2
3
�
regularization_losses
!layer_metrics
"layer_regularization_losses
#metrics
$non_trainable_variables
trainable_variables

%layers
	variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
&layer_metrics
'layer_regularization_losses
(metrics
)non_trainable_variables
trainable_variables

*layers
	variables
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
+layer_metrics
,layer_regularization_losses
-metrics
.non_trainable_variables
trainable_variables

/layers
	variables
 
 
 
�
regularization_losses
0layer_metrics
1layer_regularization_losses
2metrics
3non_trainable_variables
trainable_variables

4layers
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

50
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	6total
	7count
8	variables
9	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

60
71

8	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/layer_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/layer_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/layer_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/layer_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biaslayer_normalization/gammalayer_normalization/beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_52216
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp4Adam/layer_normalization/gamma/m/Read/ReadVariableOp3Adam/layer_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp4Adam/layer_normalization/gamma/v/Read/ReadVariableOp3Adam/layer_normalization/beta/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_52521
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biaslayer_normalization/gammalayer_normalization/beta	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense/kernel/mAdam/dense/bias/m Adam/layer_normalization/gamma/mAdam/layer_normalization/beta/mAdam/dense/kernel/vAdam/dense/bias/v Adam/layer_normalization/gamma/vAdam/layer_normalization/beta/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_52588��
�T
�
!__inference__traced_restore_52588
file_prefix0
assignvariableop_dense_kernel:	�+
assignvariableop_1_dense_bias::
,assignvariableop_2_layer_normalization_gamma:9
+assignvariableop_3_layer_normalization_beta:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: :
'assignvariableop_11_adam_dense_kernel_m:	�3
%assignvariableop_12_adam_dense_bias_m:B
4assignvariableop_13_adam_layer_normalization_gamma_m:A
3assignvariableop_14_adam_layer_normalization_beta_m::
'assignvariableop_15_adam_dense_kernel_v:	�3
%assignvariableop_16_adam_dense_bias_v:B
4assignvariableop_17_adam_layer_normalization_gamma_v:A
3assignvariableop_18_adam_layer_normalization_beta_v:
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_layer_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_layer_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_dense_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_adam_dense_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_layer_normalization_gamma_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp3assignvariableop_14_adam_layer_normalization_beta_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_layer_normalization_gamma_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_layer_normalization_beta_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19f
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_20�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
%__inference_dense_layer_call_fn_52373

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_520052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_52074

inputs
dense_52006:	�
dense_52008:'
layer_normalization_52054:'
layer_normalization_52056:
identity��dense/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52006dense_52008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_520052
dense/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_52054layer_normalization_52056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_520532-
+layer_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_520712
activation/PartitionedCall~
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_52341

inputs
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_520742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_52053

inputs+
mul_2_readvariableop_resource:)
add_readvariableop_resource:
identity��add/ReadVariableOp�mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape�
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������2	
ReshapeY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������2
ones[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������2
	Reshape_1�
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
mul_2�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_52195
input_1
dense_52183:	�
dense_52185:'
layer_normalization_52188:'
layer_normalization_52190:
identity��dense/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_52183dense_52185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_520052
dense/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_52188layer_normalization_52190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_520532-
+layer_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_520712
activation/PartitionedCall~
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
#__inference_signature_wrapper_52216
input_1
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_519882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
@__inference_dense_layer_call_and_return_conditional_losses_52005

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�E
�
@__inference_model_layer_call_and_return_conditional_losses_52272

inputs7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:?
1layer_normalization_mul_2_readvariableop_resource:=
/layer_normalization_add_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_2/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAdd|
layer_normalization/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
layer_normalization/Shape�
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack�
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1�
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul�
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack�
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1�
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x�
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1�
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0�
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape�
layer_normalization/ReshapeReshapedense/BiasAdd:output:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������2
layer_normalization/Reshape�
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/ones/packed�
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
layer_normalization/ones/Const�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������2
layer_normalization/ones�
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2"
 layer_normalization/zeros/packed�
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer_normalization/zeros/Const�
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������2
layer_normalization/zerosy
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const}
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_1�
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:2&
$layer_normalization/FusedBatchNormV3�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:���������2
layer_normalization/Reshape_1�
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization/mul_2/ReadVariableOp�
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer_normalization/mul_2�
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:*
dtype02(
&layer_normalization/add/ReadVariableOp�
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer_normalization/adds
activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/Gelu/mul/x�
activation/Gelu/mulMulactivation/Gelu/mul/x:output:0layer_normalization/add:z:0*
T0*'
_output_shapes
:���������2
activation/Gelu/mulu
activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
activation/Gelu/Cast/x�
activation/Gelu/truedivRealDivlayer_normalization/add:z:0activation/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������2
activation/Gelu/truediv�
activation/Gelu/ErfErfactivation/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������2
activation/Gelu/Erfs
activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
activation/Gelu/add/x�
activation/Gelu/addAddV2activation/Gelu/add/x:output:0activation/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������2
activation/Gelu/add�
activation/Gelu/mul_1Mulactivation/Gelu/mul:z:0activation/Gelu/add:z:0*
T0*'
_output_shapes
:���������2
activation/Gelu/mul_1t
IdentityIdentityactivation/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_52354

inputs
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_521412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_52141

inputs
dense_52129:	�
dense_52131:'
layer_normalization_52134:'
layer_normalization_52136:
identity��dense/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52129dense_52131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_520052
dense/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_52134layer_normalization_52136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_520532-
+layer_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_520712
activation/PartitionedCall~
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_activation_layer_call_fn_52441

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_520712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
a
E__inference_activation_layer_call_and_return_conditional_losses_52071

inputs
identity]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xj
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*'
_output_shapes
:���������2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/xw
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*'
_output_shapes
:���������2
Gelu/truediv_
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:���������2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xr
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:���������2

Gelu/addm

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:���������2

Gelu/mul_1b
IdentityIdentityGelu/mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_52165
input_1
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_521412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�	
a
E__inference_activation_layer_call_and_return_conditional_losses_52436

inputs
identity]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xj
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*'
_output_shapes
:���������2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/xw
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*'
_output_shapes
:���������2
Gelu/truediv_
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:���������2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xr
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:���������2

Gelu/addm

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:���������2

Gelu/mul_1b
IdentityIdentityGelu/mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_52180
input_1
dense_52168:	�
dense_52170:'
layer_normalization_52173:'
layer_normalization_52175:
identity��dense/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_52168dense_52170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_520052
dense/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_52173layer_normalization_52175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_520532-
+layer_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_520712
activation/PartitionedCall~
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
@__inference_dense_layer_call_and_return_conditional_losses_52364

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�M
�
 __inference__wrapped_model_51988
input_1=
*model_dense_matmul_readvariableop_resource:	�9
+model_dense_biasadd_readvariableop_resource:E
7model_layer_normalization_mul_2_readvariableop_resource:C
5model_layer_normalization_add_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�,model/layer_normalization/add/ReadVariableOp�.model/layer_normalization/mul_2/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense/BiasAdd�
model/layer_normalization/ShapeShapemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:2!
model/layer_normalization/Shape�
-model/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-model/layer_normalization/strided_slice/stack�
/model/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/layer_normalization/strided_slice/stack_1�
/model/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/layer_normalization/strided_slice/stack_2�
'model/layer_normalization/strided_sliceStridedSlice(model/layer_normalization/Shape:output:06model/layer_normalization/strided_slice/stack:output:08model/layer_normalization/strided_slice/stack_1:output:08model/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/layer_normalization/strided_slice�
model/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2!
model/layer_normalization/mul/x�
model/layer_normalization/mulMul(model/layer_normalization/mul/x:output:00model/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
model/layer_normalization/mul�
/model/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/model/layer_normalization/strided_slice_1/stack�
1model/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model/layer_normalization/strided_slice_1/stack_1�
1model/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model/layer_normalization/strided_slice_1/stack_2�
)model/layer_normalization/strided_slice_1StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_1/stack:output:0:model/layer_normalization/strided_slice_1/stack_1:output:0:model/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model/layer_normalization/strided_slice_1�
!model/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!model/layer_normalization/mul_1/x�
model/layer_normalization/mul_1Mul*model/layer_normalization/mul_1/x:output:02model/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2!
model/layer_normalization/mul_1�
)model/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2+
)model/layer_normalization/Reshape/shape/0�
)model/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)model/layer_normalization/Reshape/shape/3�
'model/layer_normalization/Reshape/shapePack2model/layer_normalization/Reshape/shape/0:output:0!model/layer_normalization/mul:z:0#model/layer_normalization/mul_1:z:02model/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'model/layer_normalization/Reshape/shape�
!model/layer_normalization/ReshapeReshapemodel/dense/BiasAdd:output:00model/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������2#
!model/layer_normalization/Reshape�
%model/layer_normalization/ones/packedPack!model/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2'
%model/layer_normalization/ones/packed�
$model/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2&
$model/layer_normalization/ones/Const�
model/layer_normalization/onesFill.model/layer_normalization/ones/packed:output:0-model/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������2 
model/layer_normalization/ones�
&model/layer_normalization/zeros/packedPack!model/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2(
&model/layer_normalization/zeros/packed�
%model/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%model/layer_normalization/zeros/Const�
model/layer_normalization/zerosFill/model/layer_normalization/zeros/packed:output:0.model/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������2!
model/layer_normalization/zeros�
model/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
model/layer_normalization/Const�
!model/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!model/layer_normalization/Const_1�
*model/layer_normalization/FusedBatchNormV3FusedBatchNormV3*model/layer_normalization/Reshape:output:0'model/layer_normalization/ones:output:0(model/layer_normalization/zeros:output:0(model/layer_normalization/Const:output:0*model/layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:2,
*model/layer_normalization/FusedBatchNormV3�
#model/layer_normalization/Reshape_1Reshape.model/layer_normalization/FusedBatchNormV3:y:0(model/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:���������2%
#model/layer_normalization/Reshape_1�
.model/layer_normalization/mul_2/ReadVariableOpReadVariableOp7model_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:*
dtype020
.model/layer_normalization/mul_2/ReadVariableOp�
model/layer_normalization/mul_2Mul,model/layer_normalization/Reshape_1:output:06model/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
model/layer_normalization/mul_2�
,model/layer_normalization/add/ReadVariableOpReadVariableOp5model_layer_normalization_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,model/layer_normalization/add/ReadVariableOp�
model/layer_normalization/addAddV2#model/layer_normalization/mul_2:z:04model/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layer_normalization/add
model/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/activation/Gelu/mul/x�
model/activation/Gelu/mulMul$model/activation/Gelu/mul/x:output:0!model/layer_normalization/add:z:0*
T0*'
_output_shapes
:���������2
model/activation/Gelu/mul�
model/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
model/activation/Gelu/Cast/x�
model/activation/Gelu/truedivRealDiv!model/layer_normalization/add:z:0%model/activation/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������2
model/activation/Gelu/truediv�
model/activation/Gelu/ErfErf!model/activation/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������2
model/activation/Gelu/Erf
model/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
model/activation/Gelu/add/x�
model/activation/Gelu/addAddV2$model/activation/Gelu/add/x:output:0model/activation/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������2
model/activation/Gelu/add�
model/activation/Gelu/mul_1Mulmodel/activation/Gelu/mul:z:0model/activation/Gelu/add:z:0*
T0*'
_output_shapes
:���������2
model/activation/Gelu/mul_1z
IdentityIdentitymodel/activation/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp-^model/layer_normalization/add/ReadVariableOp/^model/layer_normalization/mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2\
,model/layer_normalization/add/ReadVariableOp,model/layer_normalization/add/ReadVariableOp2`
.model/layer_normalization/mul_2/ReadVariableOp.model/layer_normalization/mul_2/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�1
�
__inference__traced_save_52521
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop?
;savev2_adam_layer_normalization_gamma_m_read_readvariableop>
:savev2_adam_layer_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop?
;savev2_adam_layer_normalization_gamma_v_read_readvariableop>
:savev2_adam_layer_normalization_beta_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop;savev2_adam_layer_normalization_gamma_m_read_readvariableop:savev2_adam_layer_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop;savev2_adam_layer_normalization_gamma_v_read_readvariableop:savev2_adam_layer_normalization_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*|
_input_shapesk
i: :	�:::: : : : : : : :	�::::	�:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�
�
%__inference_model_layer_call_fn_52085
input_1
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_520742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
3__inference_layer_normalization_layer_call_fn_52424

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_520532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�E
�
@__inference_model_layer_call_and_return_conditional_losses_52328

inputs7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:?
1layer_normalization_mul_2_readvariableop_resource:=
/layer_normalization_add_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_2/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAdd|
layer_normalization/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
layer_normalization/Shape�
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack�
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1�
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul�
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack�
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1�
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x�
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1�
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0�
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape�
layer_normalization/ReshapeReshapedense/BiasAdd:output:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������2
layer_normalization/Reshape�
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/ones/packed�
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
layer_normalization/ones/Const�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������2
layer_normalization/ones�
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2"
 layer_normalization/zeros/packed�
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer_normalization/zeros/Const�
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������2
layer_normalization/zerosy
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const}
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_1�
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:2&
$layer_normalization/FusedBatchNormV3�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:���������2
layer_normalization/Reshape_1�
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization/mul_2/ReadVariableOp�
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer_normalization/mul_2�
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:*
dtype02(
&layer_normalization/add/ReadVariableOp�
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer_normalization/adds
activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/Gelu/mul/x�
activation/Gelu/mulMulactivation/Gelu/mul/x:output:0layer_normalization/add:z:0*
T0*'
_output_shapes
:���������2
activation/Gelu/mulu
activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
activation/Gelu/Cast/x�
activation/Gelu/truedivRealDivlayer_normalization/add:z:0activation/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������2
activation/Gelu/truediv�
activation/Gelu/ErfErfactivation/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������2
activation/Gelu/Erfs
activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
activation/Gelu/add/x�
activation/Gelu/addAddV2activation/Gelu/add/x:output:0activation/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������2
activation/Gelu/add�
activation/Gelu/mul_1Mulactivation/Gelu/mul:z:0activation/Gelu/add:z:0*
T0*'
_output_shapes
:���������2
activation/Gelu/mul_1t
IdentityIdentityactivation/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_52415

inputs+
mul_2_readvariableop_resource:)
add_readvariableop_resource:
identity��add/ReadVariableOp�mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape�
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������2	
ReshapeY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������2
ones[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������2
	Reshape_1�
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
mul_2�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������>

activation0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�J
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
*B&call_and_return_all_conditional_losses
C__call__
D_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*E&call_and_return_all_conditional_losses
F__call__"
_tf_keras_layer
�
axis
	gamma
beta
regularization_losses
trainable_variables
	variables
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"
_tf_keras_layer
�
regularization_losses
trainable_variables
	variables
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"
_tf_keras_layer
�
iter

beta_1

beta_2
	decay
 learning_ratem:m;m<m=v>v?v@vA"
	optimizer
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
regularization_losses
!layer_metrics
"layer_regularization_losses
#metrics
$non_trainable_variables
trainable_variables

%layers
	variables
C__call__
D_default_save_signature
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
,
Kserving_default"
signature_map
:	�2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
&layer_metrics
'layer_regularization_losses
(metrics
)non_trainable_variables
trainable_variables

*layers
	variables
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2layer_normalization/gamma
&:$2layer_normalization/beta
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
+layer_metrics
,layer_regularization_losses
-metrics
.non_trainable_variables
trainable_variables

/layers
	variables
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
0layer_metrics
1layer_regularization_losses
2metrics
3non_trainable_variables
trainable_variables

4layers
	variables
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	6total
	7count
8	variables
9	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
60
71"
trackable_list_wrapper
-
8	variables"
_generic_user_object
$:"	�2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*2 Adam/layer_normalization/gamma/m
+:)2Adam/layer_normalization/beta/m
$:"	�2Adam/dense/kernel/v
:2Adam/dense/bias/v
,:*2 Adam/layer_normalization/gamma/v
+:)2Adam/layer_normalization/beta/v
�2�
@__inference_model_layer_call_and_return_conditional_losses_52272
@__inference_model_layer_call_and_return_conditional_losses_52328
@__inference_model_layer_call_and_return_conditional_losses_52180
@__inference_model_layer_call_and_return_conditional_losses_52195�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_model_layer_call_fn_52085
%__inference_model_layer_call_fn_52341
%__inference_model_layer_call_fn_52354
%__inference_model_layer_call_fn_52165�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_51988input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_52364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_52373�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_layer_normalization_layer_call_and_return_conditional_losses_52415�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_layer_normalization_layer_call_fn_52424�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_activation_layer_call_and_return_conditional_losses_52436�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_activation_layer_call_fn_52441�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_52216input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_51988r1�.
'�$
"�
input_1����������
� "7�4
2

activation$�!

activation����������
E__inference_activation_layer_call_and_return_conditional_losses_52436X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� y
*__inference_activation_layer_call_fn_52441K/�,
%�"
 �
inputs���������
� "�����������
@__inference_dense_layer_call_and_return_conditional_losses_52364]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� y
%__inference_dense_layer_call_fn_52373P0�-
&�#
!�
inputs����������
� "�����������
N__inference_layer_normalization_layer_call_and_return_conditional_losses_52415\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
3__inference_layer_normalization_layer_call_fn_52424O/�,
%�"
 �
inputs���������
� "�����������
@__inference_model_layer_call_and_return_conditional_losses_52180h9�6
/�,
"�
input_1����������
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_52195h9�6
/�,
"�
input_1����������
p

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_52272g8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_52328g8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������
� �
%__inference_model_layer_call_fn_52085[9�6
/�,
"�
input_1����������
p 

 
� "�����������
%__inference_model_layer_call_fn_52165[9�6
/�,
"�
input_1����������
p

 
� "�����������
%__inference_model_layer_call_fn_52341Z8�5
.�+
!�
inputs����������
p 

 
� "�����������
%__inference_model_layer_call_fn_52354Z8�5
.�+
!�
inputs����������
p

 
� "�����������
#__inference_signature_wrapper_52216}<�9
� 
2�/
-
input_1"�
input_1����������"7�4
2

activation$�!

activation���������