import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-full text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:pointer-events-none",
  {
    variants: {
      variant: {
        primary: "bg-aa-red hover:bg-aa-red-dark text-white shadow-glow-red hover:shadow-none hover:scale-[0.98]",
        secondary: "bg-aa-blue hover:bg-[#0b6296] text-white",
        ghost: "glass glass-hover text-white/70 hover:text-white",
        outline: "border border-white/15 text-white/70 hover:text-white hover:border-white/30 hover:bg-white/5",
        accent: "bg-aa-blue-light/15 hover:bg-aa-blue-light/25 text-aa-blue-light border border-aa-blue-light/30",
      },
      size: {
        sm: "h-8 px-4 text-xs",
        md: "h-10 px-6",
        lg: "h-12 px-8 text-base",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  )
);
Button.displayName = "Button";

export { Button, buttonVariants };
