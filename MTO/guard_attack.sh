# usage: bash guard_attack.sh <mode> <startIndex> <generate_device_id> <guard_device_id> [dataset] [extra args, e.g. --guard]
# --paraphrase-iters is set huge by default: query re-paraphrasing needs an external LLM API (see evaluate_util.harmful_paraphrase)
# run from the MTO/ directory with the project venv active; models are read from the HF cache.
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/data1/huggingface/hub}
export HF_HUB_OFFLINE=1
mode=$1
startIndex=$2
generate_device_id=$3
guard_device_id=$4
dataset=${5:-advbench_original}
shift $(( $# < 5 ? $# : 5 ))

python cold_decoding.py  \
	--seed 12 \
	--mode $mode \
	--dataset $dataset \
	--pretrained_model Llama-3.1-8b-instruct \
	--pretrained_guard_model d3-flash \
	--device-id $generate_device_id \
	--guard-device-id $guard_device_id \
	--init-temp 1 \
    --length 20 \
	--max-length 20 \
	--num-iters 200 \
	--min-iters 0 \
	--goal-weight 100 \
    --rej-weight 100 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start $startIndex \
	--end 100 \
	--lr-nll-portion 1.0 \
    --topk 10 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,200,500,1500\
	--large_gs_std  1.0,0.5,0.1,0.01  \
	--stepsize-ratio 1  \
    --batch-size 1 \
    --print-every 1000 \
	--fp16 \
	--paraphrase-iters 1000000 \
	"$@"
