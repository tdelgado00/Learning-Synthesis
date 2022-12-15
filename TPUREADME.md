# Accelerated training and inference enviroment setup guide

## Warning: before doing this you must have gcloud correctly installed in your computer and configured in the project:

``` gcloud init ```


We can use 3 types of TPU VMs for free during the trial period

    1. 5 on-demand Cloud TPU v2-8 device(s) in zone us-central1-f
    2. 100 preemptible Cloud TPU v2-8 device(s) in zone us-central1-f
    3. 5 on-demand Cloud TPU v3-8 device(s) in zone europe-west4-a
 
## Create a Cloud TPU VM for each one respectively, run:
	1. ``` bash tpu-vm-creation-ondemand-us-v2-pytorch.sh```
	2. ``` bash tpu-vm-creation-preemptive-us-v2-pytorch.sh```
	3. ``` bash tpu-vm-creation-ondemand-europe-v3-pytorch.sh```


## ssh into the TPU vm, use the following template command (remember to adapt it to either 1 or 2 or 3):

	```gcloud compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project <project-id>```


## Listing your Cloud TPU resources

``` gcloud compute tpus tpu-vm list --zone=<zone> ```

## Retrieving information about your Cloud TPU

``` gcloud compute tpus tpu-vm describe <tpu-name> --zone=<zone>  ```

## Stopping your Cloud TPU resources

gcloud compute tpus tpu-vm stop <tpu-name> --zone=<zone>


## Starting your Cloud TPU resources

```gcloud compute tpus tpu-vm start <tpu-name> --zone  <zone> ```

## Deleting your VM and Cloud TPU resources

``` gcloud compute tpus tpu-vm delete <tpu-name> --zone=<zone>```

The deletion might take several minutes. Verify the resources have been deleted by running gcloud compute tpus list --zone=${ZONE}.

# Once inside the VM

Set the XRT TPU device configuration:

``` export XRT_TPU_CONFIG="localservice;0;localhost:51011" ```
``` unset LD_PRELOAD ```

### Then you can use the VM as a normal computer that has access to a TPU service. remember that you have to explicitly set the TPU usage from the scripts where you run the programs. 


