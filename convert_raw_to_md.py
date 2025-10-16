from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple
import subprocess
from openai import OpenAI
import tempfile

from datetime import datetime

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "raw_file.txt"
NOTE_DIR = ROOT / "note_file"
PIC_DIR = ROOT / "pic_file"

# 从deepseek_api.txt 读取DeepSeek API 密钥
with open(ROOT / "deepseek_api.txt") as f:
    DEEPSEEK_API_KEY = f.read().strip()

# 生成当前时间戳，用于 md 文件名和图片子目录，例如 20251013_153045
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 输出 md 名称示例: 笔记20251013_153045.md
# 输出 md 名称示例: 笔记20251013_153045.md
MD_FILENAME = f"笔记{TIMESTAMP}.md"
# 在 pic 目录下创建按时间戳分组的子目录
PIC_SUBDIR = PIC_DIR / TIMESTAMP
def find_image_paths(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """返回在文本中找到的图片路径及其字符串范围。

    返回值列表中每项是 (path, (start_index, end_index))，path 为原始匹配字符串中的路径部分。
    """
    results: List[Tuple[str, Tuple[int, int]]] = []

    # markdown 图片语法 ![alt](path)
    md_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for m in md_pattern.finditer(text):
        path = m.group(1).strip()
        # 使用整个匹配范围，便于替换整个 ![alt](path)
        results.append((path, (m.start(), m.end())))

    # HTML <img src="path" />
    html_pattern = re.compile(r"<img\s+[^>]*src=[\'\"]([^\'\"]+)[\'\"][^>]*>")
    for m in html_pattern.finditer(text):
        path = m.group(1).strip()
        # 使用整个匹配范围，便于替换整个 <img ...>
        results.append((path, (m.start(), m.end())))

    # bare paths (lines that look like paths) - a simple heuristic: lines containing / or \\ and an image extension
    # 改进的裸路径检测：匹配非空白且不含常见分隔符的字符串，且以图片扩展名结尾
    path_line_pattern = re.compile(r"(^|\s)([^\s\)\"\'<>]+?\.(?:png|jpg|jpeg|gif|svg))(\s|$)", re.IGNORECASE | re.MULTILINE)
    for m in path_line_pattern.finditer(text):
        path = m.group(2).strip()
        results.append((path, (m.start(2), m.end(2))))

    # 去重，按发现顺序
    seen = set()
    ordered: List[Tuple[str, Tuple[int, int]]] = []
    for p, rng in results:
        if p not in seen:
            seen.add(p)
            ordered.append((p, rng))
    return ordered

def copy_image(src: str, dest_dir: Path) -> Tuple[Path, bool]:
    """复制图片 src 到 dest_dir，返回 (dest_path, True) 如果成功，否则返回 (Path(src), False).

    src 可以是绝对路径或相对路径（相对到 ROOT）。
    """
    src_path = Path(src)
    if not src_path.is_absolute():
        # 相对到项目根
        src_path = (ROOT / src_path).resolve()

    if not src_path.exists():
        return (Path(src), False)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 准备目标文件名：为 png 源优先生成 .jpg 目标名（以实现较好压缩），同时生成原始后缀的回退目标名
    base_name = src_path.stem
    orig_suffix = src_path.suffix.lower()

    def unique_path(name: str, suffix: str) -> Path:
        candidate = dest_dir / f"{name}{suffix}"
        if not candidate.exists():
            return candidate
        i = 1
        while True:
            candidate = dest_dir / f"{name}-{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1

    # 定义两个可能的目标路径：jpg 版本（用于 png->jpg）和原始后缀版本（回退用）
    dest_path_jpg = unique_path(base_name, ".jpg")
    dest_path_orig = unique_path(base_name, orig_suffix)
    # 默认 dest_path（如果不是 png，则用原始后缀；如果为 png，则先尝试 jpg）
    if orig_suffix == ".png":
        dest_path = dest_path_jpg
    else:
        dest_path = dest_path_orig

    # 对于 jpg/jpeg/png，尝试使用 ffmpeg 压缩为宽高各减半（面积约为 1/4）
    if src_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        try:
            # 检查 ffmpeg 是否可用
            from shutil import which

            ffmpeg_path = which("ffmpeg")
            if ffmpeg_path:
                suffix = orig_suffix
                # 临时文件，用于存放中间 jpg 文件或最终输出
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    # 如果源是 PNG，先转为临时 JPG
                    if suffix == ".png":
                        tmp_jpg = td_path / (src_path.stem + ".jpg")
                        cmd1 = [ffmpeg_path, "-y", "-i", str(src_path), str(tmp_jpg)]
                        p1 = subprocess.run(cmd1, capture_output=True)
                        if p1.returncode != 0 or not tmp_jpg.exists():
                            print(f"png->jpg 转换失败，ffmpeg stderr: {p1.stderr.decode(errors='ignore')}")
                            raise RuntimeError("png->jpg 转换失败")
                        work_src = tmp_jpg
                    else:
                        work_src = src_path

                    # 对 work_src（应为 jpg）执行缩放并降低质量，输出到临时 jpg
                    tmp_out = td_path / (dest_path_jpg.name)
                    cmd2 = [ffmpeg_path, "-y", "-i", str(work_src), "-vf", "scale=iw/2:ih/2", "-q:v", "10", str(tmp_out)]
                    p2 = subprocess.run(cmd2, capture_output=True)
                    if p2.returncode == 0 and tmp_out.exists():
                        # 比较 tmp_out 大小与原始文件大小（原始可能为 png）
                        try:
                            orig_size = src_path.stat().st_size
                        except Exception:
                            orig_size = None
                        try:
                            tmp_size = tmp_out.stat().st_size
                        except Exception:
                            tmp_size = None

                        if orig_size is not None and tmp_size is not None and tmp_size > orig_size:
                            # 压缩后的 jpg 比原始 png 大 -> 回退为复制原始 png（保留 png）
                            print(f"压缩后文件变大（{tmp_size} > {orig_size}），回退到复制原始 PNG。")
                            # 将原始 png 复制到回退目标（dest_path_orig）并返回该路径
                            shutil.copy2(str(src_path), str(dest_path_orig))
                            return (dest_path_orig, True)
                        else:
                            # tmp_out 是合格的 jpg，移动到目标 jpg 路径
                            shutil.move(str(tmp_out), str(dest_path_jpg))
                            return (dest_path_jpg, True)
                    else:
                        print(f"ffmpeg 压缩失败，stderr: {p2.stderr.decode(errors='ignore')}")
                        raise RuntimeError("ffmpeg 压缩失败")
            else:
                print("警告: 未找到 ffmpeg，可选的图片压缩将被跳过，使用普通复制。")
        except Exception as e:
            print(f"调用 ffmpeg 时发生异常，回退到普通复制: {e}")

    # 默认直接复制（或在 ffmpeg 不可用/失败时）
    try:
        shutil.copy2(src_path, dest_path)
        return (dest_path, True)
    except Exception as e:
        print(f"复制文件失败: {e}")
        return (Path(src), False)

def convert(raw_path: Path, note_dir: Path, pic_dir: Path) -> Path:
    text = raw_path.read_text(encoding="utf-8")

    # 提前创建并确定写入 md 的子目录（用于后续相对路径计算）
    note_subdir = note_dir / TIMESTAMP
    note_subdir.mkdir(parents=True, exist_ok=True)

    images = find_image_paths(text)

    # 替换规则：从后向前替换以便索引不受影响
    for path_text, (start, end) in sorted(images, key=lambda x: x[1][0], reverse=True):
        # 清理可能的引号
        p = path_text.strip('"\'')
        # 处理 file:// 前缀（如 file://E:\path\to\file.jpg 或 file:///C:/path）
        if p.lower().startswith('file://'):
            # 去掉前缀
            p = p[7:]
            # 如果以三个斜杠开始（file:///C:/...），也移除多余的斜杠
            if p.startswith('///'):
                p = p[3:]
            # 在 windows 上，可能以 /C:/ 开头，去掉前导斜杠
            if os.name == 'nt' and p.startswith('/') and len(p) > 2 and p[2] == ':':
                p = p[1:]
        # 处理 URL：如果是 http(s) 链接则不复制
        if re.match(r"^https?://", p, re.IGNORECASE):
            # 将链接替换为 md 内嵌图片（保留在线链接），使用完整匹配范围替换
            md_img = f"![image]({p})"
            text = text[:start] + md_img + text[end:]
            continue

        dest_path, ok = copy_image(p, PIC_SUBDIR)
        if ok:
            # 生成相对路径（相对于 md 文件所在的 note_subdir）
            try:
                rel = os.path.relpath(dest_path, start=note_subdir)
            except Exception:
                rel = dest_path.name
            # 在 f-string 外先替换反斜杠以避免语法错误
            rel_fixed = rel.replace('\\', '/')
            md_img = f"![image]({rel_fixed})"
            # 用新文本替换原始匹配范围
            text = text[:start] + md_img + text[end:]
        else:
            print(f"警告: 找不到图片文件: {p}，保留原始文本。")

    # 确保 note_dir 下的时间戳子目录存在，并写入 md 文件（使用时间戳命名）
    note_subdir = note_dir / TIMESTAMP
    note_subdir.mkdir(parents=True, exist_ok=True)
    out_md = note_subdir / MD_FILENAME
    out_md.write_text(text, encoding="utf-8")
    # 生成 AI提示词：基于 raw_file.txt，删除时间戳、把尖括号包裹的图片替换为 [图片]
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
        # 根据你的说明：源文件的时间戳格式是 MM-DD HH:MM:SS（例如 10-09 23:48:14）
        # 因此只删除此类时间戳，保留发言人文本
        # 匹配示例形式： 10-09 23:48:14
        time_pattern = re.compile(r"\b\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b")
        cleaned = time_pattern.sub("", raw_text)
        # 将尖括号包裹的图片链接 <...> 全部替换为 [图片]
        cleaned = re.sub(r"<[^>]+>", "[图片]", cleaned)
        # 另外，如果存在 markdown 图片语法或内联图片链接也用 [图片] 表示（可选，但常见）
        cleaned = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "[图片]", cleaned)
        # 去掉时间段删除后行尾多余空格，并压缩多重空行
        cleaned = re.sub(r"[ \t]+(?=\n)", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # 生成关于标题的AI提示词
        prompt_header = (
            "请为下面的对话内容生成一个简洁的标题，指出其讨论的主要内容。\n"
            "只需返回一个标题，不要包含额外注释。\n\n"
        )
        title_prompt = prompt_header + cleaned.strip()
        (note_subdir / "title_prompt.txt").write_text(title_prompt, encoding="utf-8")
        # 生成关于摘要的AI提示词
        prompt_header = (
            "请为下面的对话内容生成一个简洁的摘要，"
            "总结主要观点和结论，以md文档的形式返还给我\n"
            "返回的内容以\"# 摘要\"开头,不需要在开头添加```markdown,或在结尾添加```\n"
            "在对话中,发言人'总是懒得改名字'和'泠酱''泠天阁'这两个昵称都是同一个人, 请统一用'泠酱'来称呼他\n\n"
            "如果对话中涉及到具体的作品名称,请单独在摘要中以一行列出这些作品名称,并用**加粗**标记\n\n"
            )
        summary_prompt = prompt_header + cleaned.strip()
        (note_subdir / "summary_prompt.txt").write_text(summary_prompt, encoding="utf-8")
    except Exception as e:
        print(f"生成 title_prompt.txt 时出错: {e}")
    return out_md

def deepseek_generate_title(prompt_dir: Path) -> Tuple[str, str]:
    # 读入title_prompt.txt, 调用deepseek API 获取标题和摘要
    title_prompt = (prompt_dir / "title_prompt.txt").read_text(encoding="utf-8")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": title_prompt},
        ],
        stream=False
    )
    title = response.choices[0].message.content.strip()
    # 读入summary_prompt.txt, 调用deepseek API 获取摘要
    summary_prompt = (prompt_dir / "summary_prompt.txt").read_text(encoding="utf-8")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": summary_prompt},
        ],
        stream=False
    )
    summary = response.choices[0].message.content.strip()
    return title, summary

# 为生成好的md文档调整格式,使其外观类似于线上聊天,并且与deepseek输出的格式相仿
# 具体调整包括:
# - 在文档开头添加 "# 正文" 标记
# - 在每个"发言人:时间戳"行前添加 "## " 标记
# - 对发言人加粗显示
# 将"总是懒得改名字"替换为"泠天阁"
def format_md(md_path: Path):
        """格式化 md 文件：
        - 在文件开头确保有 "# 正文" 标题
        - 将形如 "发言人: 时间戳..." 的行改为 "### **发言人**: 时间戳..."
        - 如果前后两条消息的发言人相同，且时间戳相差不到10分钟，则删除后一条消息的“发言人：时间戳”行
        """
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"读取 {md_path} 失败: {e}")
            return

        lines = text.splitlines()
        out_lines: List[str] = []

        # 在文档顶端添加 # 正文（如果尚未存在）
        if not text.lstrip().startswith("# 正文"):
            out_lines.append("# 正文")
            out_lines.append("")

        speaker_line_re = re.compile(r"^\s*([^:\n]+?)\s*:\s*(.+)$")
        # 匹配时间戳并捕获后续消息内容
        timestamp_split_re = re.compile(r"^\s*(?P<ts>(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2}))\s*(?P<content>.*)$")

        from datetime import datetime, timedelta

        prev_speaker = None
        prev_dt = None

        def parse_timestamp(ts_str: str) -> datetime | None:
            ts_str = ts_str.strip()
            now = datetime.now()
            for fmt in ("%Y-%m-%d %H:%M:%S", "%m-%d %H:%M:%S", "%H:%M:%S"):
                try:
                    dt = datetime.strptime(ts_str, fmt)
                    # 对于没有年份的格式，填充当前年份
                    if fmt == "%m-%d %H:%M:%S":
                        dt = dt.replace(year=now.year)
                    if fmt == "%H:%M:%S":
                        dt = dt.replace(year=now.year, month=now.month, day=now.day)
                    return dt
                except Exception:
                    continue
            return None

        for line in lines:
            # 标准化已经以 '## ' 或 '### ' 开头的行，去掉前缀后继续解析
            stripped = line.lstrip()
            if stripped.startswith("## "):
                candidate = stripped[3:]
            elif stripped.startswith("### "):
                candidate = stripped[5:]
            else:
                candidate = line

            m = speaker_line_re.match(candidate)
            if m:
                speaker = m.group(1).strip()
                rest = m.group(2).strip()
                # 如果右侧以时间戳开头，拆分时间戳和剩余内容
                m_ts = timestamp_split_re.match(rest)
                if m_ts:
                    ts_str = m_ts.group("ts")
                    content_after = m_ts.group("content").strip()
                    curr_dt = parse_timestamp(ts_str)

                    # 如果与上一条相同发言人且时间差小于10分钟，则省略本行的发言人头，仅保留内容
                    if prev_speaker == speaker and prev_dt and curr_dt:
                        delta = abs((curr_dt - prev_dt).total_seconds())
                        if delta < 10 * 60:
                            if content_after:
                                out_lines.append(content_after)
                            # 不更新时间戳，保留 prev_dt 以便连续多条合并
                            continue

                    # 否则输出新的发言人头（使用 ###）
                    header = f"### **{speaker}**: {ts_str}"
                    if content_after:
                        header = header + (" " + content_after)
                    out_lines.append(header)
                    prev_speaker = speaker
                    if curr_dt:
                        prev_dt = curr_dt
                    continue

            # 非发言人行或未包含时间戳，直接追加并重置 prev_dt? 保留 prev_speaker，但不改变 prev_dt
            out_lines.append(line)

        new_text = "\n".join(out_lines)
        # 保持文件末尾的换行
        if text.endswith("\n") and not new_text.endswith("\n"):
            new_text += "\n"
        # 将"总是懒得改名字"替换为"泠天阁"
        new_text = new_text.replace("总是懒得改名字", "泠天阁")
        try:
            md_path.write_text(new_text, encoding="utf-8")
        except Exception as e:
            print(f"写入 {md_path} 失败: {e}")

def main():
    # 检测是否有命令行参数,若有参数"-manual"则不从剪贴板读取而是打开raw_file.txt等候用户输入
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "-manual":
        print("手动模式: 请编辑 raw_file.txt 后保存，然后关闭此窗口以继续处理。")
        # 确保 raw 文件存在
        if not RAW_FILE.exists():
            RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
            RAW_FILE.write_text("", encoding="utf-8")

        # 清空文件
        RAW_FILE.write_text("", encoding="utf-8")

        try:
            initial_mtime = RAW_FILE.stat().st_mtime
        except Exception:
            initial_mtime = None

        # 打开文件以便用户编辑
        try:
            if os.name == "nt":
                # 在 Windows 上使用默认程序打开
                os.startfile(str(RAW_FILE))
        except Exception as e:
            print("打开 raw_file.txt 时出错，请手动打开并编辑:", RAW_FILE, e)

        import time

        print("请在打开的编辑器中编辑 raw_file.txt，保存后脚本将继续执行。")
        waited = 0
        timeout = 60 * 30  # 最多等待 30 分钟
        poll_interval = 1.0
        try:
            # 等待直到检测到文件被保存（mtime 变化），然后主动尝试关闭占用该文件的编辑器窗口
            saved = False
            while waited < timeout:
                try:
                    mtime = RAW_FILE.stat().st_mtime
                except Exception:
                    mtime = None
                if initial_mtime is None:
                    if mtime is not None:
                        saved = True
                else:
                    if mtime is not None and mtime != initial_mtime:
                        saved = True

                if saved:
                    # 已检测到保存；不再尝试关闭编辑器窗口或终止进程，直接继续执行脚本
                    break

                time.sleep(poll_interval)
                waited += poll_interval
        except KeyboardInterrupt:
            print("检测到用户中断，退出。")
            return

        if waited >= timeout:
            print("警告: 等待超时30 分钟，继续执行脚本。")
    else:
        # 尝试从系统剪贴板读取文本并覆盖 raw_file.txt（仅在剪贴板非空时）
        try:
            # 使用 PowerShell 的 Get-Clipboard（适用于 Windows）
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"], capture_output=True, text=True)
            clip_text = proc.stdout
            if clip_text and clip_text.strip():
                RAW_FILE.write_text(clip_text, encoding="utf-8")
                print(f"已用剪贴板内容覆盖: {RAW_FILE}")
        except Exception as e:
            print(f"从剪贴板读取失败，继续使用现有 {RAW_FILE}（错误: {e}）")

        if not RAW_FILE.exists():
            print(f"找不到 {RAW_FILE}，请确认当前目录为脚本所在目录并且 raw_file.txt 存在。")
            return

    # 生成md文件
    out = convert(RAW_FILE, NOTE_DIR, PIC_DIR)
    print(f"已生成: {out}")
    # 对md文件进行格式化
    try:
        format_md(out)
    except Exception as e:
        print(f"格式化 md 文件时出错: {e}")

    # 调用deepseek API 生成笔记标题
    try:
        title, summary = deepseek_generate_title(out.parent)
        # 修改md文件的文件名
        new_md_path = out.parent / f"{title}.md"
        out.rename(new_md_path)
        # 在md文件的开头部分添加摘要以及三行换行
        with new_md_path.open("r", encoding="utf-8") as f:
            original_content = f.read()
        with new_md_path.open("w", encoding="utf-8") as f:
            f.write(f"{summary}\n\n\n{original_content}")
        # 修改父文件夹名 在原本的文件夹名称,即时间戳之前添加标题 同时更新 new_md_path
        parent_dir = out.parent
        new_parent_dir = parent_dir.parent / f"{title}_{parent_dir.name}"
        parent_dir.rename(new_parent_dir)
        new_md_path = new_parent_dir / new_md_path.name
        # 删除title_prompt.txt 和 summary_prompt.txt
        (new_parent_dir / "summary_prompt.txt").unlink(missing_ok=True)
        (new_parent_dir / "title_prompt.txt").unlink(missing_ok=True)
    except Exception as e:
        print(f"调用 DeepSeek API 生成标题时出错: {e}")
    
    # 最后清空raw_file.txt
    try:
        RAW_FILE.write_text("", encoding="utf-8")
    except Exception as e:
        print(f"清空 {RAW_FILE} 时出错: {e}")
    
if __name__ == "__main__":
    main()
