import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

/**
 * 剑指 Offer 练习题
 * 牛客网 https://www.nowcoder.net/ta/coding-interviews
 */

public class Solution {

    public static void main(String[] args) {
        System.out.println("GOOD LUCK!");
    }

    class TreeNode { //定义二叉树数据结构
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }
    class ListNode { //定义链表数据结构
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    /**
     * Q1 二维数组中查找
     * 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     */
    public boolean Find(int target, int[][] array) {
        for (int[] a : array) {
            for (int b : a) {
                if (b == target) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Q2 替换空格
     * 请实现一个函数，将一个字符串中的空格替换成“%20”。
     * 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     */
    public String replaceSpace(StringBuffer str) {
        return str.toString().replace(" ", "%20");
    }

    /**
     * Q3 从尾到头打印链表
     * 输入一个链表，从尾到头打印链表每个节点的值。
     */
    //递归方法 需 import java.util.ArrayList;
    private ArrayList<Integer> arrayListFromTailToHead = new ArrayList();
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode != null) {
            printListFromTailToHead(listNode.next);
            arrayListFromTailToHead.add(listNode.val);
        }
        return arrayListFromTailToHead;
    }
    //利用栈 需 import java.util.ArrayList;import java.util.Stack;
    public ArrayList<Integer> printListFromTailToHead_2(ListNode listNode) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (listNode==null) return arrayList;
        Stack<Integer> stack = new Stack<>();
        while (listNode!=null) {
            stack.add(listNode.val);
            listNode=listNode.next;
        }
        while (!stack.isEmpty()) {
            arrayList.add(stack.pop());
        }
        return arrayList;
    }

    /**
     * Q4 重建出该二叉树
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        return construct(pre, 0, pre.length - 1, in, 0, in.length - 1);
    }
    public TreeNode construct(int[] pre, int ps, int pe, int[] in, int is, int ie) {
        if (ps > pe || is > ie) {
            return null;
        }
        TreeNode root = new TreeNode(pre[ps]);
        int sep = 0;
        for (int i = is; i <= ie; i++) {
            if (in[i] == root.val) {
                sep = i;
                break;
            }
        }
        root.left = construct(pre, ps + 1, ps + sep - is, in, is, sep - 1);
        root.right = construct(pre, ps + 1 + sep - is, pe, in, sep + 1, ie);
        return root;
    }

    /**
     * Q5 用两个栈实现队列
     * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
     */
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    public void push(int node) {
        stack1.push(node);
    }
    public int pop() {
        while (!stack1.isEmpty()) {
            stack2.push(stack1.pop());
        }
        int result = stack2.pop();
        while (!stack2.isEmpty()) {
            stack1.push(stack2.pop());
        }
        return result;
    }

    /**
     * Q6 旋转数组的最小数字
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     */
    public int minNumberInRotateArray(int[] array) {
        if (array.length == 0) {
            return 0;
        }
        int temp = array[0];
        for (int num : array) {
            if (num < temp) {
                temp = num;
            }
        }
        return temp;
    }

    /**
     * Q7 斐波那契数列
     * 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。
     * n<=39
     *
     * f(n) = f(n-1) + f(n-2)
     */
    public int Fibonacci(int n) {
        if (n == 1 || n == 2) {
            return 1;
        } else if (n == 0) {
            return 0;
        } else {
            int a = 1, b = 1;
            int temp = 0;
            for (int i = 0; i < n - 2; i++) {
                temp = a + b;
                a = b;
                b = temp;
            }
            return temp;
        }
    }

    /**
     * Q8 跳台阶
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     *
     * 规律：f(n) = f(n-1) + f(n-2)，类似斐波那契数列
     */
    //循环解
    public int JumpFloor_1(int target) {
        int a = 1, b = 2;
        if (target == 1) {
            return a;
        } else if (target == 2) {
            return b;
        } else {
            int temp = 0;
            for (int i = 2; i < target; i++) {
                temp = a + b;
                a = b;
                b = temp;
            }
            return temp;
        }
    }
    //递归解
    public int JumpFloor_2(int target) {
        if (target == 2) {
            return 2;
        } else if (target == 1) {
            return 1;
        } else {
            return JumpFloor_2(target - 1) + JumpFloor_2(target - 2);
        }
    }

    /**
     * Q9 变态跳台阶
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     *
     * f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1)
     * f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
     * 可以得出：f(n) = 2*f(n-1)
     */
    public int JumpFloorII(int target) {
        if (target == 0) {
            return 0;
        } else if (target == 1) {
            return 1;
        } else {
            return 2 * JumpFloorII(target - 1);
        }
    }

    /**
     * Q10 矩形覆盖
     * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
     * 请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     *
     * 画图分析规律 f(n) = f(n-1) + f(n-2) 同青蛙跳
     */
    public int RectCover(int target) {
        if (target == 0) {
            return 0;
        } else if (target == 1) {
            return 1;
        } else if (target == 2) {
            return 2;
        } else {
            return RectCover(target - 1) + RectCover(target - 2);
        }
    }

    /**
     * Q11 二进制中1的个数
     * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     *
     * Integer.toBinaryString() 输出为二进制补码
     */
    public int NumberOf1(int n) {
        int num = 0;
        char[] chars = Integer.toBinaryString(n).toCharArray();
        for (int i = 0; i < chars.length; i++) {
            if (chars[i] == '1') {
                num++;
            }
        }
        return num;
    }

    /**
     * Q12 数值到整数次方
     * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     */
    //Java函数封装求解
    public double Power_1(double base, int exponent) {
        return Math.pow(base, exponent);
    }
    //算法求解
    public double Power_2(double base, int exponent) {
        if (exponent < 0) {
            double temp = 1;
            for (int i = 0; i < -exponent; i++) {
                temp = temp * base;
            }
            return 1 / temp;
        } else if (exponent > 0) {
            double temp = 1;
            for (int i = 0; i < exponent; i++) {
                temp = temp * base;
            }
            return temp;
        } else {
            return 1;
        }
    }
    //简单优化
    public double Power_3(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        } else {
            double temp = 1;
            for (int i = 0; i < Math.abs(exponent); i++) {
                temp = temp * base;
            }
            if (exponent < 0) {
                return 1 / temp;
            }
            return temp;
        }
    }

    /**
     * Q13 调整数组顺序，使奇数位于偶数前面
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
     * 使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，
     * 并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     *
     * 类似冒泡算法，前偶后奇就交换
     */
    public void reOrderArray(int[] array) {
        for (int i = 0; i < array.length - 1; i++) {
            for (int j = 0; j < array.length - 1 - i; j++) {
                if (array[j] % 2 == 0 && array[j + 1] % 2 == 1) {
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }

    /**
     * Q14 链表中倒数第K个节点
     * 输入一个链表，输出该链表中倒数第k个结点。
     */
    public ListNode FindKthToTail_1(ListNode head, int k) {
        if (head == null) {
            return head;
        }
        int size = 0;
        ListNode temp = head;
        while (temp != null) {
            size++;
            temp = temp.next;
        }
        if (size < k) {
            return null;
        }
        int num = size - k;
        for (int i = 0; i < num; i++) {
            head = head.next;
        }
        return head;
    }
    //看到的优雅方法 一次遍历即可
    public ListNode FindKthToTail_2(ListNode head, int k) {
        ListNode p, q;
        p = q = head;
        int i = 0;
        for (; p != null; i++) {
            if (i >= k)
                q = q.next;
            p = p.next;
        }
        return i < k ? null : q;
    }

    /**
     * Q15 反转链表
     * 输入一个链表，反转链表后，输出链表的所有元素。
     *
     * 设置前节点，后节点，当前节点，一次调换顺序，一次遍历后，pre为最后节点，也就是反转后的第一个节点
     */
    public ListNode ReverseList(ListNode head) {
        if (head==null)
            return null;
        ListNode pre = null;
        ListNode next = null;
        while (head!=null) {
            next=head.next;
            head.next=pre;
            pre=head;
            head=next;
        }
        return pre;
    }

    /**
     * Q16 合并两个排序的链表
     *输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     */
    public ListNode Merge(ListNode list1,ListNode list2) {
        if(list1==null){
            return list2;
        }else if(list2==null){
            return list1;
        }
        ListNode mergeList = null;
        if (list1.val<list2.val) {
            mergeList=list1;
            mergeList.next=Merge(list1.next,list2);
        } else {
            mergeList=list2;
            mergeList.next=Merge(list1,list2.next);
        }
        return mergeList;
    }

    /**
     * Q17 树的子结构
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     *
     *
     *
     *
     *
     * ??? 大佬答案占位
     *
     *
     *
     *
     *
     */
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if(root2==null) return false;
        if(root1==null && root2!=null) return false;
        boolean flag = false;
        if(root1.val==root2.val){
            flag = isSubTree(root1,root2);
        }
        if(!flag){
            flag = HasSubtree(root1.left, root2);
            if(!flag){
                flag = HasSubtree(root1.right, root2);
            }
        }
        return flag;
    }
    private boolean isSubTree(TreeNode root1, TreeNode root2) {
        if(root2==null) return true;
        if(root1==null && root2!=null) return false;
        if(root1.val==root2.val){
            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
        }else{
            return false;
        }
    }

    /**
     * Q18 二叉树的镜像
     * 操作给定的二叉树，将其变换为源二叉树的镜像。
     */
    //递归
    public void Mirror(TreeNode root) {
        if (root==null) return;
        TreeNode temp = root.left;
        root.left=root.right;
        root.right=temp;
        if (root.left!=null) Mirror(root.left);
        if (root.right!=null) Mirror(root.right);
    }
    //非递归 需 import java.util.Stack;
    public void Mirror_2(TreeNode root) {
        if(root == null) return;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.push(root);
        while(!stack.empty()) {
            TreeNode node = stack.pop();
            if(node.left != null || node.right != null) {
                TreeNode temp = node.left;
                node.left = node.right;
                node.right = temp;
            }
            if(node.left != null) stack.push(node.left);
            if(node.right != null) stack.push(node.right);
        }
    }

    /**
     * Q19 顺时针打印矩阵
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
     * 例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
     * 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
     *
     * 依据圈数打印，一圈包含两行两列，循环打印。计算上下左右四个边界，完整打印一圈，边界内缩一圈，继续打印。
     */
    //需 import java.util.ArrayList;
    //循环实现
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix==null || matrix.length==0) return result;
        int row = matrix.length;
        int col = matrix[0].length;
        int left = 0, top = 0, right = col - 1, bottom = row - 1;
        while (left<=right && top<=bottom) {
            for (int i=left;i<=right;i++) result.add(matrix[top][i]);
            for (int i=top+1;i<=bottom;i++) result.add(matrix[i][right]);
            if (top!=bottom) for (int i=right-1;i>=left;i--) result.add(matrix[bottom][i]);
            if (left!=right) for (int i=bottom-1;i>top;i--) result.add(matrix[i][left]);
            top++;bottom--;left++;right--;
        }
        return result;
    }
    //递归实现
    public ArrayList<Integer> printMatrix_2(int [][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix==null || matrix.length==0) return result;
        printMatrixClockwise(matrix,0,matrix[0].length-1,0,matrix.length-1,result);
        return result;
    }
    public void printMatrixClockwise
            (int [][] matrix,int left,int right,int top,int bottom,ArrayList<Integer> result) {
        if (left<=right && top<=bottom) {
            for (int i=left;i<=right;i++) result.add(matrix[top][i]);
            for (int i=top+1;i<=bottom;i++) result.add(matrix[i][right]);
            if (top!=bottom) for (int i=right-1;i>=left;i--) result.add(matrix[bottom][i]);
            if (left!=right) for (int i=bottom-1;i>top;i--) result.add(matrix[i][left]);
            printMatrixClockwise(matrix,left+1,right-1,top+1,bottom-1,result);
        } else {
            return;
        }
    }

    /**
     * Q20 包含main函数的栈
     *定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
     */
    //需 import java.util.Stack; import java.util.Iterator;
    Stack<Integer> stack = new Stack<>();
//    public void push(int node) {
//        stack.push(node);
//    }
//    public void pop() {
//        stack.pop();
//    }
//    public int top() {
//        return stack.peek();
//    }
    public int min() {
        Stack<Integer> tempStack = new Stack<>();
        int min = Integer.MAX_VALUE;
        while (!stack.empty()){
            int temp = stack.pop();
            tempStack.push(temp);
            if (min>temp){
                min = temp;
            }
        }
        while (!tempStack.empty()) {
            stack.push(tempStack.pop());
        }
        return min;
    }

    /**
     * Q21 栈的压入，弹出顺序
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
     * 假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列对应的一个弹出序列，
     * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
     *
     * 建立辅助栈遍历压栈顺序，对比给定的弹出顺序，若有则出栈，最后判断辅助栈是否为空，空为真，非空为假
     */
    //需 import java.util.Stack;
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        if (pushA==null || popA==null || pushA.length==0 || popA.length==0 || pushA.length!=popA.length) {
            return false;
        }
        Stack<Integer> temp = new Stack<>();
        int flag = 0;
        for (int i=0;i<pushA.length;i++) {
            temp.push(pushA[i]);
            while (!temp.empty() && temp.peek()==popA[flag]) {
                temp.pop();
                flag++;
            }
        }
        return temp.empty();
    }

    /**
     * Q22 从上往下打印二叉树
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     *
     * 层序遍历 一般利用队列实现
     */
    //需 import java.util.ArrayList; import java.util.Queue; import java.util.LinkedList;
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (root==null) return arrayList;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode tempNode = queue.remove();
            arrayList.add(tempNode.val);
            if (tempNode.left!=null) queue.add(tempNode.left);
            if (tempNode.right!=null) queue.add(tempNode.right);
        }
        return arrayList;
    }
    //使用ArrayList实现Deque 需 import java.util.ArrayList;
    public ArrayList<Integer> PrintFromTopToBottom_2(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (root==null) return arrayList;
        ArrayList<TreeNode> queue = new ArrayList<>();
        queue.add(root);
        while (queue.size()!=0) {
            TreeNode tempNode = queue.remove(0);
            if (tempNode.left!=null) queue.add(tempNode.left);
            if (tempNode.right!=null) queue.add(tempNode.right);
            arrayList.add(tempNode.val);
        }
        return arrayList;
    }

    /**
     * Q23 二叉搜索树的后序遍历序列
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
     *
     * 注意二叉搜索树的性质是左节点小于右节点
     */
//    public boolean VerifySquenceOfBST(int [] sequence) {
//
//    }
}
