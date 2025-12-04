# Prompt Engineering - Kỹ Thuật Thiết Kế Câu Lệnh AI

## Giới thiệu

Prompt Engineering là kỹ thuật thiết kế và tối ưu hóa các câu lệnh (prompts) để tương tác hiệu quả với các mô hình AI như GPT, Claude, Gemini. Đây là kỹ năng quan trọng để khai thác tối đa khả năng của AI trong thực tế.

## Tại sao Prompt Engineering quan trọng?

- Quyết định chất lượng output của AI
- Giảm chi phí API calls (ít phải retry)
- Tiết kiệm thời gian xử lý
- Kiểm soát tốt hơn hành vi của model
- Không cần fine-tuning vẫn có kết quả tốt

## Cấu trúc Prompt cơ bản

```
[Context] → [Instruction] → [Input/Question] → [Output Format]
```

**Ví dụ:**

```
Bạn là chuyên gia marketing với 10 năm kinh nghiệm.
Hãy viết 3 tiêu đề quảng cáo sáng tạo cho sản phẩm: Nước hoa hồng dưỡng da.
Mỗi tiêu đề không quá 10 từ, tập trung vào lợi ích cho da.
```

## Các kỹ thuật Prompt cơ bản

### 1. Clear and Specific (Rõ ràng & Cụ thể)

❌ **Không tốt:**

```
Viết về AI
```

✅ **Tốt:**

```
Viết một đoạn văn 150 từ giải thích cách AI transformer hoạt động,
dành cho sinh viên năm 2 ngành CNTT, sử dụng ví dụ cụ thể.
```

### 2. Role Prompting (Gán vai trò)

```
Bạn là một [role] với [experience/expertise].
Nhiệm vụ của bạn là [task].
```

**Ví dụ:**

- "Bạn là Python developer senior..."
- "Bạn là giáo viên toán..."
- "Bạn là chuyên gia UX/UI..."

### 3. Few-Shot Learning (Học từ ví dụ)

Cung cấp ví dụ input-output để AI hiểu pattern:

```
Phân loại sentiment của các câu sau:

Câu: "Sản phẩm tuyệt vời!"
Sentiment: Tích cực

Câu: "Giao hàng chậm quá."
Sentiment: Tiêu cực

Câu: "Chất lượng ổn, giá hơi cao."
Sentiment: ?
```

### 4. Chain-of-Thought (CoT) - Tư duy theo chuỗi

Yêu cầu AI giải thích từng bước:

```
Hãy giải bài toán sau từng bước một:
"Một cửa hàng giảm giá 20% cho sản phẩm 500k, sau đó giảm thêm 10%.
Giá cuối cùng là bao nhiêu?"

Hãy:
1. Tính giá sau lần giảm đầu
2. Tính giá sau lần giảm thứ hai
3. Đưa ra kết quả cuối
```

### 5. Constrain Output (Giới hạn output)

```
Viết một email xin việc:
- Độ dài: 200-250 từ
- Giọng văn: Chuyên nghiệp nhưng thân thiện
- Bao gồm: Giới thiệu, kỹ năng, lý do apply
- Không dùng: Từ "passion", "dedicated"
```

## Kỹ thuật nâng cao

### 1. Self-Consistency

Yêu cầu AI đưa ra nhiều cách giải, sau đó chọn câu trả lời phổ biến nhất.

```
Giải bài toán này bằng 3 cách khác nhau,
sau đó so sánh và đưa ra đáp án chính xác nhất.
```

### 2. Tree of Thoughts (ToT)

Khám phá nhiều nhánh reasoning khác nhau:

```
Hãy phân tích vấn đề này theo 3 góc độ:
1. Góc độ kỹ thuật
2. Góc độ người dùng
3. Góc độ kinh doanh

Sau đó tổng hợp để đưa ra giải pháp tối ưu.
```

### 3. ReAct (Reasoning + Acting)

Kết hợp suy luận và hành động:

```
Nhiệm vụ: Tìm thông tin về giá Bitcoin hôm nay

Suy nghĩ: Tôi cần tìm kiếm giá Bitcoin real-time
Hành động: Tìm kiếm "Bitcoin price today"
Quan sát: [kết quả tìm kiếm]
Suy nghĩ: Giá hiện tại là $X
Đáp án: [tổng hợp thông tin]
```

### 4. Meta Prompting

Yêu cầu AI tự cải thiện prompt:

```
Hãy phân tích prompt sau và đề xuất cách cải thiện:
"Viết code Python"

Đánh giá các khía cạnh: clarity, specificity, context, constraints.
```

## Các Pattern Prompt phổ biến

### Pattern 1: Task Decomposition

```
Nhiệm vụ phức tạp: [task]

Hãy chia thành các bước nhỏ:
1. [Sub-task 1]
2. [Sub-task 2]
...

Sau đó thực hiện từng bước.
```

### Pattern 2: Persona Pattern

```
Tôi muốn bạn đóng vai [persona].
Bạn nên [behavior/characteristic].
Khi tôi hỏi về [topic], hãy trả lời theo cách [style].
```

### Pattern 3: Template Pattern

```
Tạo [output type] theo template sau:

**Tiêu đề:** [...]
**Mở đầu:** [...]
**Nội dung chính:**
- Điểm 1: [...]
- Điểm 2: [...]
**Kết luận:** [...]
```

### Pattern 4: Reflection Pattern

```
Sau khi hoàn thành task, hãy:
1. Review output của bạn
2. Tìm điểm yếu hoặc thiếu sót
3. Cải thiện và đưa ra version tốt hơn
```

## Best Practices

### ✅ Nên làm:

1. **Specific over vague**: "Viết 5 câu về Python" thay vì "Nói về Python"
2. **Provide context**: Càng nhiều context, output càng relevant
3. **Specify format**: JSON, table, bullet points, markdown...
4. **Use delimiters**: Dùng ```, """, ### để phân tách rõ ràng
5. **Iterate**: Test và cải thiện prompt liên tục
6. **Give examples**: Few-shot learning rất hiệu quả
7. **Set constraints**: Độ dài, tone, style, restrictions

### ❌ Không nên:

1. Quá mơ hồ: "Làm cái gì đó với data này"
2. Quá phức tạp: Nhiều tasks trong 1 prompt
3. Mâu thuẫn: Yêu cầu "ngắn gọn" nhưng "chi tiết"
4. Thiếu context: AI không biết background
5. Không test: Chỉ viết 1 lần rồi dùng luôn

## Ví dụ thực tế

### Ví dụ 1: Code Generation

```
Viết Python function để:
- Đọc file CSV
- Xử lý missing values bằng mean imputation
- Chuẩn hóa các cột numeric về range [0,1]
- Return DataFrame đã xử lý

Yêu cầu:
- Sử dụng pandas
- Include error handling
- Add docstring và type hints
- Viết 2-3 test cases
```

### Ví dụ 2: Content Creation

```
Viết bài blog post về "Machine Learning cho beginners"

Yêu cầu:
- Độ dài: 800-1000 từ
- Giọng điệu: Thân thiện, dễ hiểu
- Cấu trúc:
  * Hook hấp dẫn (2-3 câu)
  * Giải thích ML là gì (simple terms)
  * 3 ứng dụng thực tế
  * 5 bước bắt đầu học ML
  * Call-to-action
- Include: 1-2 analogies để giải thích khái niệm
- Tránh: Thuật ngữ quá kỹ thuật
```

### Ví dụ 3: Data Analysis

```
Dataset: [upload CSV]

Hãy phân tích data theo các bước:
1. Describe cấu trúc dataset (rows, columns, types)
2. Tìm missing values và outliers
3. Phân tích correlation giữa các features
4. Đưa ra 3 insights chính từ data
5. Recommend preprocessing steps

Output format: Markdown với tables và bullet points
```

## Tools & Resources

**Testing Prompts:**

- ChatGPT Playground
- Claude.ai
- PromptPerfect
- Anthropic Console

**Learning Resources:**

- OpenAI Prompt Engineering Guide
- Anthropic Prompt Library
- PromptingGuide.ai
- Learn Prompting

**Prompt Libraries:**

- Awesome ChatGPT Prompts
- FlowGPT
- PromptBase
- ShareGPT

## Tips nâng cao

### 1. Prompt Chaining

Chia task lớn thành nhiều prompts nhỏ, output của prompt này là input của prompt sau.

### 2. Temperature Control

- Temperature thấp (0-0.3): Deterministic, consistent
- Temperature cao (0.7-1.0): Creative, diverse

### 3. System Prompts

Sử dụng system message để set behavior tổng thể:

```
System: "Bạn là trợ lý lập trình chuyên nghiệp, luôn cung cấp code sạch và documented."
User: "Viết function sort array"
```

### 4. Negative Prompting

Nói rõ những gì KHÔNG muốn:

```
Viết mô tả sản phẩm.
KHÔNG sử dụng: hyperbole, buzzwords, emoji, all caps.
```

## Common Mistakes & Solutions

| Mistake              | Problem                | Solution                        |
| -------------------- | ---------------------- | ------------------------------- |
| Prompt quá ngắn      | Output generic         | Thêm context và constraints     |
| Không specify format | Output không dùng được | Yêu cầu format cụ thể           |
| Quá nhiều tasks      | AI confused            | Chia nhỏ thành multiple prompts |
| Thiếu examples       | AI không hiểu pattern  | Dùng few-shot learning          |
| Không iterate        | Stuck với prompt tệ    | Test và improve liên tục        |

## Kết luận

Prompt Engineering là kỹ năng thiết yếu trong thời đại AI. Key points:

- **Clarity is king**: Prompt càng rõ ràng, output càng tốt
- **Context matters**: Cung cấp đủ context cho AI
- **Iterate**: Không có prompt hoàn hảo từ lần đầu
- **Learn from examples**: Study prompts của người khác
- **Practice**: Thử nghiệm với nhiều styles khác nhau

Prompt tốt = Output tốt = Tiết kiệm thời gian & tiền bạc!

---

**Pro tip cuối:** Lưu lại các prompts hiệu quả vào một prompt library riêng để tái sử dụng! 🚀
