# B3/B4 更新日志

## 2026-07-12

- 修复 B3 在服务器整合包 code 目录下运行时的导入路径问题。
- 修复 B3 的 get_tools_schema、execute_tool_calls 与 B1 整合调用的位置参数兼容问题。
- 修复 B4 的 generate_ai_message 与 B1 六参数调用方式不兼容的问题。
- 修复 B4 返回格式，统一返回 status、ai_message、raw_model_output、error。
- 修复 B4 mock 模式下 file_reader 参数与 tools_schema 不一致的问题。
- 已在服务器 AgentB 环境通过 B1 mock 模式和 prompt_json 真实模型模式测试。
