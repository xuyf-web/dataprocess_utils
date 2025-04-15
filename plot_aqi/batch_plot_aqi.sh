#!/bin/bash

# 设置日期范围
START_DATE="2024-08-02"  # 开始日期，格式：YYYY-MM-DD
END_DATE="2024-08-03"    # 结束日期，格式：YYYY-MM-DD

# 设置输出目录
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# 获取当前时间作为日志文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_plot_${TIMESTAMP}.log"

# 记录开始时间
echo "Batch plot started at $(date)" | tee -a ${LOG_FILE}
echo "Date range: ${START_DATE} to ${END_DATE}" | tee -a ${LOG_FILE}
echo "----------------------------------------" | tee -a ${LOG_FILE}

# 将开始日期转换为时间戳
current_date=$(date -d "${START_DATE}" +%s)
end_timestamp=$(date -d "${END_DATE}" +%s)

# 循环处理每一天
while [ ${current_date} -le ${end_timestamp} ]
do
    # 将时间戳转换回日期格式
    current_date_str=$(date -d @${current_date} +"%Y%m%d")
    
    echo "Processing date: ${current_date_str}" | tee -a ${LOG_FILE}
    
    # 调用Python脚本并记录输出
    python3.12 plot_aqi_oneday.py ${current_date_str} >> ${LOG_FILE} 2>&1
    
    # 检查Python脚本的执行状态
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${current_date_str}" | tee -a ${LOG_FILE}
    else
        echo "Error processing ${current_date_str}" | tee -a ${LOG_FILE}
    fi
    
    echo "----------------------------------------" | tee -a ${LOG_FILE}
    
    # 增加一天
    current_date=$((current_date + 86400))
done

# 记录结束时间
echo "Batch plot completed at $(date)" | tee -a ${LOG_FILE}
echo "Log file saved to: ${LOG_FILE}" 